import unittest
import os
from unittest.mock import Mock

from linker.main import link_texts, write_generator
class CustomDict(dict):
    def __setattr__(self, key, value):
        self.__dict__[key] = value

from addict import Dict

def mock_search(query, k, search_type):
    mock_data = {
        'test': Dict({
            'score': 0.9,
            'metadata': {'file_path': '/path/of/test'}
        }),
        'linking': Dict({
            'score': 0.95,
            'metadata': {'file_path': '/path/of/linking'}
        }),
        'linking function': Dict({
            'score': 0.9,
            'metadata': {'file_path': '/path/of/linking_function'}
        }),
        'function test': Dict({
            'score': 0.85,
            'metadata': {'file_path': '/path/of/function_test'}
        }),
        'threshold': Dict({
            'score': 0.75,
            'metadata': {'file_path': '/path/of/threshold'}
        })
    }
    return [mock_data.get(query, Dict(metadata={'file_path': "RESULT" }))]
class TestLinkTexts(unittest.TestCase):

    def setUp(self):
        self.temp_dir = "temp_test_dir"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.mock_vector_store = Mock()

    def tearDown(self):
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(self.temp_dir)

    def test_no_links(self):
        sample_text = "This is a sample sentence."  + "".join([" a"] * 1000)
        with open(os.path.join(self.temp_dir, "test_no_links.md"), "w") as f:
            f.write(sample_text)

        self.mock_vector_store.search.return_value = []

        link_texts(self.temp_dir, self.mock_vector_store)

        with open(os.path.join(self.temp_dir, "test_no_links.md"), "r") as f:
            linked_text = f.read()

        self.assertEqual(linked_text, sample_text)

    def test_multiple_links(self):
        sample_text = "This is a test sentence for the linking function." + "".join([" a"] * 1000)
        with open(os.path.join(self.temp_dir, "test_multiple_links.md"), "w") as f:
            f.write(sample_text)



        self.mock_vector_store.search.side_effect = mock_search

        write_generator(
            link_texts(self.temp_dir, self.mock_vector_store)
        )

        with open(os.path.join(self.temp_dir, "test_multiple_links.md"), "r") as f:
            linked_text = f.read()

        self.assertIn("[test](/path/of/test)", linked_text)
        self.assertIn("[linking](/path/of/linking)", linked_text)

    def test_overlapping_links(self):
        sample_text = "This is a linking function test."  + "".join([" a"] * 1000)
        with open(os.path.join(self.temp_dir, "test_overlapping_links.md"), "w") as f:
            f.write(sample_text)



        self.mock_vector_store.search.side_effect = mock_search

        link_texts(self.temp_dir, self.mock_vector_store)

        with open(os.path.join(self.temp_dir, "test_overlapping_links.md"), "r") as f:
            linked_text = f.read()

        self.assertIn("[linking function](/path/of/linking_function)", linked_text)
        self.assertNotIn("[function test](/path/of/function_test)", linked_text)

    def test_thresholding(self):
        sample_text = "This is a threshold test."
        with open(os.path.join(self.temp_dir, "test_threshold.md"), "w") as f:
            f.write(sample_text)



        self.mock_vector_store.search.side_effect = mock_search

        link_texts(self.temp_dir, self.mock_vector_store, relevance_threshold=0.8)

        with open(os.path.join(self.temp_dir, "test_threshold.md"), "r") as f:
            linked_text = f.read()

        self.assertNotIn("[threshold](/path/of/threshold)", linked_text)

if __name__ == "__main__":
    unittest.main()
