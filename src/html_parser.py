import os
from bs4 import BeautifulSoup
from src.schemas import OLAMExtraction, Hyperlink, SectionResult

class OLAMHTMLParser:
    """
    A specialized parser for Cigna OLAM HTML documents.
    It identifies Medical and Pharmacy sections and extracts text points
    along with associated hyperlink metadata.
    """
    def __init__(self):
        """Initializes the parser with target section headers and document extensions."""
        self.sections_to_extract = ["Medical:", "Pharmacy:"]
        # Include .html and .htm as valid document/form extensions as requested
        self.doc_extensions = [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".html", ".htm", ".rtf", ".msg", ".ppt", ".pptx"]

    def _get_contextual_title(self, a_tag):
        """Extracts a concise title for a link by looking at its surrounding context."""
        # Get the full text of the surrounding block (e.g., <li> or <p>)
        block = a_tag.find_parent(["li", "p", "div", "td"])
        if block:
            full_context = " ".join(block.get_text(separator=" ", strip=True).split())
            return full_context if len(full_context) < 300 else full_context[:297] + "..."
        return a_tag.get_text(strip=True)

    def _extract_points_and_mapped_docs(self, container) -> SectionResult:
        """
        Recursively traverses a container tag to extract text points and map them
        to any document links found within the same context.
        """
        final_points, hyperlinks, seen_urls = [], [], set()
        
        # Helper to decide if an a-tag is a "document"
        def is_document_link(href):
            is_doc = any(href.lower().endswith(ext) for ext in self.doc_extensions)
            is_olam_form = "/OLAMForms/" in href
            return is_doc or is_olam_form

        def add_point(text, a_tags=[]):
            clean_text = " ".join(text.replace("\u00a0", " ").split())
            if not clean_text: return
            final_points.append(clean_text)
            for a in a_tags:
                href = a.get("href", "")
                if href and is_document_link(href) and href not in seen_urls:
                    hyperlinks.append(Hyperlink(text=clean_text, url=href))
                    seen_urls.add(href)

        def get_all_a_tags(element):
            return element.find_all("a", recursive=True)

        def build_text_with_links(element):
            parts = []
            for child in element.children:
                if isinstance(child, str):
                    parts.append(child)
                elif child.name == "a":
                    link_text = child.get_text(strip=True)
                    href = child.get("href", "")
                    if link_text and href:
                        if not (link_text.startswith("http") or link_text.startswith("www")):
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
                if elem.name == "br": return
                has_sub_blocks = elem.find(split_tags, recursive=True) is not None
                if not has_sub_blocks:
                    add_point(build_text_with_links(elem), get_all_a_tags(elem))
                else:
                    current_text_parts = []
                    current_a_tags = []
                    for child in elem.children:
                        if isinstance(child, str):
                            current_text_parts.append(child)
                        elif child.name in split_tags:
                            if current_text_parts:
                                add_point("".join(current_text_parts), current_a_tags)
                                current_text_parts, current_a_tags = [], []
                            if child.name != "br": process_element(child)
                        else:
                            current_text_parts.append(build_text_with_links(child))
                            current_a_tags.extend(get_all_a_tags(child))
                    if current_text_parts:
                        add_point("".join(current_text_parts), current_a_tags)
            elif elem.name == "a":
                 add_point(build_text_with_links(elem), [elem])
            else:
                has_sub_blocks = elem.find(split_tags, recursive=True) is not None
                if not has_sub_blocks:
                    add_point(build_text_with_links(elem), get_all_a_tags(elem))
                else:
                    for child in elem.children: process_element(child)

        for child in container.children:
            process_element(child)
        
        # Deduplicate while preserving order
        deduped_points = []
        seen_texts = set()
        for text in final_points:
            if text not in seen_texts:
                deduped_points.append(text)
                seen_texts.add(text)

        return SectionResult(
            text=deduped_points,
            hyperlinks=hyperlinks,
            has_link=(len(hyperlinks) > 0)
        )

    def parse_file(self, file_path: str) -> OLAMExtraction:
        """Reads and parses a single OLAM HTML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        extraction = OLAMExtraction()

        for section_name in self.sections_to_extract:
            target_td = soup.find(lambda tag: tag.name == "td" and 
                                tag.get_text().strip().startswith(section_name))
            
            if target_td:
                content_td = target_td.find_next_sibling("td")
                if content_td:
                    result = self._extract_points_and_mapped_docs(content_td)
                    if "Medical" in section_name:
                        extraction.medical = result
                    elif "Pharmacy" in section_name:
                        extraction.pharmacy = result
        
        return extraction

def process_olam_html(file_path: str) -> OLAMExtraction:
    """Wrapper function to instantiate the parser and extract data from a file."""
    parser = OLAMHTMLParser()
    return parser.parse_file(file_path)
