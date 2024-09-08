import os
from io import StringIO
from tqdm import tqdm

MAX_SIZE = 3 * 1024 * 1024 * 1024 # 2GB
HEADER = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/ http://www.mediawiki.org/xml/export-0.10.xsd" version="0.10" xml:lang="en">'

def read_and_segment(input_file_path, output_file_paths, max_size=MAX_SIZE):
    os.makedirs(output_file_paths, exist_ok=True)

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        current_size = 0
        current_line_number = 0
        segment_number = 1

        output_file = open(output_file_paths + str(segment_number) + '.xml', 'w', encoding='utf-8')

        for line in tqdm(input_file, desc="Processing", unit="line"):
            current_size += len(line)
            current_line_number += 1

            output_file.write(line)

            if current_size >= max_size and ('</page>' in line):
                output_file.write('</mediawiki>')

                segment_number += 1

                # reset tracking the file
                output_file.close()
                output_file = open(output_file_paths + str(segment_number) + '.xml', 'w', encoding='utf-8')
                output_file.write(HEADER)
                current_size = len(HEADER)

read_and_segment('./dataset/enwiki-latest-pages-articles.xml', './dataset/segments/')
