import numpy as np
import unittest

# 问题: https://www.deep-ml.com/problems/1
# 相关视频：https://www.bilibili.com/video/BV1Wy4y1h7ii/?spm_id_from=333.337.search-card.all.click&vd_source=ad9baad7255ec872c204d39c3e5ba4d8
# 相关资料：https://www.labri.fr/perso/nrougier/from-python-to-numpy/
def matrix_dot_vector(a:list[list[float]], b:list[float]) -> list[float]:
    return  np.matmul(a, b).tolist()


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    
    return np.mean(matrix, axis=0 if mode == 'column' else 1).tolist()    

class TestDeepML(unittest.TestCase):
    def test_matrix_dot_vector(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [7, 8, 9]
        expected = [50, 122]
        result = matrix_dot_vector(a, b)
        self.assertEqual(result, expected, 'matrix dot vector')
        

        
    def test_calculate_matrix_mean(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected_column = [4.0, 5.0, 6.0]
        expected_row = [2.0, 5.0, 8.0]
        result_column = calculate_matrix_mean(matrix, 'column')
        result_row = calculate_matrix_mean(matrix, 'row')
        self.assertEqual(result_column, expected_column, 'column mean')
        self.assertEqual(result_row, expected_row, 'row mean')
def main():
    unittest.main()


if __name__ == "__main__":
    main()
