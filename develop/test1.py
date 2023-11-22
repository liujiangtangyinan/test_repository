

# 编写测试类，继承自unittest.TestCase
class TestExample(unittest.TestCase):

    # 编写测试方法，以'test_'开头命名
    def test_add_numbers(self):
        # 执行测试断言
        self.assertEqual(2 + 3, 5)

        # 可以使用其他断言方法进行验证
        self.assertTrue(10 > 5)
        self.assertIn('hello', 'hello world')

if __name__ == '__main__':
    unittest.main()