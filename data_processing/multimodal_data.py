import os
import shutil
from typing import List

from config import text_embedding_model, image_embedding_model
from unstructured.partition.pdf import partition_pdf

from data_processing.utils import categorize_elements, tokenize_text, save_pdf_pages_as_images
from data_processing.summarizer import generate_text_summaries, generate_table_summaries, generate_image_summaries

class MultimodalESGData:
    def __init__(self, pdf_path: str):

        self.pdf_path = pdf_path

        #initialize properties
        self.texts = []
        self.tables = []
        self.images_directory = ""

        self.text_summaries = []
        self.table_summaries = []
        self.image_summaries = []

        self.text_embedding_model = text_embedding_model
        self.image_embedding_model = image_embedding_model

        #fetch and save data
        self.extract_pdf_data()
        self.generate_summaries()
        self.erase_saved_images()

    @property
    def texts(self) -> List[str]:
        return self._texts

    @texts.setter
    def texts(self, value: List[str]):
        self._texts = value

    @property
    def tables(self) -> List[dict]:
        return self._tables

    @tables.setter
    def tables(self, value: List[dict]):
        self._tables = value

    @property
    def images_directory(self) -> str:
        return self._images_directory

    @images_directory.setter
    def images_directory(self, value: str):
        self._images_directory = value

    @property
    def text_summaries(self) -> List[str]:
        return self._text_summaries

    @text_summaries.setter
    def text_summaries(self, value: List[str]):
        self._text_summaries = value

    @property
    def table_summaries(self) -> List[str]:
        return self._table_summaries

    @table_summaries.setter
    def table_summaries(self, value: List[str]):
        self._table_summaries = value

    @property
    def image_summaries(self) -> List[str]:
        return self._image_summaries

    @image_summaries.setter
    def image_summaries(self, value: List[str]):
        self._image_summaries = value

    @property
    def images(self) -> List[str]:
        return self._images

    @images.setter
    def images(self, value: List[str]):
        self._images = value


    def extract_pdf_data(self):
        '''
        Given path to a pdf file; 
        extract and return the texts (tokenized), tables and images (path to directory)

        '''

        directory, filename = os.path.split(self.pdf_path)
        output_path = os.path.join(directory, filename.split(".")[0])

        save_pdf_pages_as_images(self.pdf_path, output_path)

        #Extract images, tables, and chunk text from a PDF file.
        raw_pdf_elements = partition_pdf(
            filename=self.pdf_path,
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=40,
            new_after_n_chars=38,
            combine_text_under_n_chars=20,
            image_output_dir_path=output_path,
        )

        texts, tables = categorize_elements(raw_pdf_elements)
        texts_4k_token = tokenize_text(texts)
        
        self.texts = texts_4k_token
        self.tables = tables
        self.images_directory = output_path

    def generate_summaries(self):
        
        # Get summaries
        self.text_summaries = generate_text_summaries(self.texts, self.text_embedding_model)
        self.table_summaries = generate_table_summaries(self.tables, self.text_embedding_model)
        self.images, self.image_summaries = generate_image_summaries(self.images_directory, self.image_embedding_model)


    def erase_saved_images(self):
        if os.path.exists(self.images_directory):
            shutil.rmtree(self.images_directory)