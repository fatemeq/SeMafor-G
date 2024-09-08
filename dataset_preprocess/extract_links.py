# coding=utf-8
import xml.etree.ElementTree as ET
import re
import json
from tqdm import tqdm

# os.makedirs(output_directory, exist_ok=True)

internal_link_pattern = r'\[\[(?:(([^:\]\[]]*:)*)([^#:|\]\[]*)(#[^|\]\[]+)?(\|[^\]\[]*)?)\]\]'

external_link_pattern = r'(https?)://([^\s\]\[]+)(\S|\||\])'

namespace = {'mediawiki': 'http://www.mediawiki.org/xml/export-0.10/'}

def extract_external_links(text):
    matches = re.findall(external_link_pattern, text)

    urls = set()
    for match in matches:
        # scheme = match[0]
        url = match[1]

        urls.add(url)

    return list(urls)

def extract_internal_links(text):
    matches = re.findall(internal_link_pattern, text)

    links = set()
    for match in matches:
        # ns = match[0]
        link = match[2]
        # hashtag = match[3]
        # title = match[4]
        links.add(link)

    return list(links)

def extract_links(text):
    return extract_internal_links(text) + extract_external_links(text)

def read_xml():
    for sid in range(30, 31):
        tree = ET.parse(f'./dataset/segments/{sid}.xml')
        root = tree.getroot()

        with open(f'./dataset/links/{sid}.json', 'w', encoding='utf-8') as input_file:
            # Iterate through each page in the XML
            for page in tqdm(root.findall('.//mediawiki:page', namespace), desc="Processing", unit="doc"):
                article_text_len = int(page.find('mediawiki:revision/mediawiki:text', namespace).get('bytes'))
                if article_text_len < 500:
                    continue

                # Extract article ID and title
                id = page.find('mediawiki:id', namespace).text

                page_content = ET.tostring(page, encoding='utf-8').decode()
                links = extract_links(page_content)

                input_file.write(json.dumps({ 'id': id, 'links': links }))
                input_file.write('\n')

read_xml()
