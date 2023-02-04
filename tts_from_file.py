import re
import subprocess

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def tts(text, chunk_number):
    subprocess.run(['tts', '--text', text, '--out_path', f'./out/chunk_{chunk_number}.wav'])

def process_text(text):
    # Remove non-alphanumeric characters (except for .?!), replacing them with spaces
    text = re.sub(r'[^\w\.\?!]+', ' ', text)

    # Split text into sentences
    sentences = re.split(r'(?<=[.?!])\s+', text)

    # Iterate over the sentences and call the tts function
    for chunk_number, sentence in enumerate(sentences):
        tts(sentence, chunk_number)

file_path = './in/README.md'
text = read_file(file_path)
process_text(text)
