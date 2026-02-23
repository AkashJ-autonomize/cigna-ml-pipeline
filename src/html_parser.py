import os
from typing import List
from bs4 import BeautifulSoup
from src.schemas import OLAMExtraction, Hyperlink, SectionResult


class OLAMHTMLParser:
    """Parses Cigna OLAM HTML files to extract Medical and Pharmacy sections."""

    def __init__(self):
        self.sections_to_extract = ["Medical:", "Pharmacy:"]
        self.doc_extensions = [
            ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".html", ".htm", ".rtf", ".msg", ".ppt", ".pptx",
        ]

    def _extract_points_and_mapped_docs(self, container) -> SectionResult:
        """Recursively traverses a section container, extracting text points and linked documents."""
        final_points, hyperlinks, seen_urls = [], [], set()

        def is_document_link(href: str) -> bool:
            return any(href.lower().endswith(ext) for ext in self.doc_extensions) or "/OLAMForms/" in href

        def add_point(text: str, a_tags: List = []):
            clean = " ".join(text.replace("\u00a0", " ").split())
            if not clean:
                return
            final_points.append(clean)
            for a in a_tags:
                href = a.get("href", "")
                if href and is_document_link(href) and href not in seen_urls:
                    hyperlinks.append(Hyperlink(text=clean, url=href))
                    seen_urls.add(href)

        def get_all_a_tags(element):
            return element.find_all("a", recursive=True)

        def build_text_with_links(element) -> str:
            parts = []
            for child in element.children:
                if isinstance(child, str):
                    parts.append(child)
                elif child.name == "a":
                    link_text = child.get_text(strip=True)
                    href = child.get("href", "")
                    if link_text and href and not (link_text.startswith("http") or link_text.startswith("www")):
                        parts.append(f"{link_text} ({href})")
                    else:
                        parts.append(link_text)
                elif child.name == "br":
                    parts.append("\n")
                else:
                    parts.append(build_text_with_links(child))
            return "".join(parts)

        def process_element(elem):
            if isinstance(elem, str):
                add_point(elem)
                return

            split_tags = ["li", "p", "div", "h1", "h2", "h3", "h4", "ul", "ol", "table"]
            if elem.name in split_tags:
                if elem.name == "br":
                    return
                has_sub_blocks = elem.find(split_tags, recursive=True) is not None
                if not has_sub_blocks:
                    add_point(build_text_with_links(elem), get_all_a_tags(elem))
                else:
                    text_parts, a_tags = [], []
                    for child in elem.children:
                        if isinstance(child, str):
                            text_parts.append(child)
                        elif child.name in split_tags:
                            if text_parts:
                                add_point("".join(text_parts), a_tags)
                                text_parts, a_tags = [], []
                            if child.name != "br":
                                process_element(child)
                        else:
                            text_parts.append(build_text_with_links(child))
                            a_tags.extend(get_all_a_tags(child))
                    if text_parts:
                        add_point("".join(text_parts), a_tags)
            elif elem.name == "a":
                add_point(build_text_with_links(elem), [elem])
            else:
                has_sub_blocks = elem.find(split_tags, recursive=True) is not None
                if not has_sub_blocks:
                    add_point(build_text_with_links(elem), get_all_a_tags(elem))
                else:
                    for child in elem.children:
                        process_element(child)

        for child in container.children:
            process_element(child)

        # Deduplicate while preserving order
        seen_texts, deduped = set(), []
        for text in final_points:
            if text not in seen_texts:
                deduped.append(text)
                seen_texts.add(text)

        return SectionResult(text=deduped, hyperlinks=hyperlinks, has_link=bool(hyperlinks))

    def parse_file(self, file_path: str) -> OLAMExtraction:
        """Reads and parses a single OLAM HTML file."""
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        extraction = OLAMExtraction()
        for section_name in self.sections_to_extract:
            target_td = soup.find(
                lambda tag: tag.name == "td" and tag.get_text().strip().startswith(section_name)
            )
            if not target_td:
                continue
            content_td = target_td.find_next_sibling("td")
            if not content_td:
                continue
            result = self._extract_points_and_mapped_docs(content_td)
            if "Medical" in section_name:
                extraction.medical = result
            elif "Pharmacy" in section_name:
                extraction.pharmacy = result

        return extraction


def process_olam_html(file_path: str) -> OLAMExtraction:
    """Entry point: parse an OLAM HTML file and return structured extraction."""
    return OLAMHTMLParser().parse_file(file_path)