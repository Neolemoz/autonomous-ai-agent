import unittest

class TestSample(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    unittest.main()
