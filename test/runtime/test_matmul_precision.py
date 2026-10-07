import unittest
from tinygrad import Context, Device, Tensor, dtypes


class TestMatmulPrecision(unittest.TestCase):
  def test_fallback_products(self):
    # Products and every FP32 partial sum are exact, independent of reduction order.
    cases = ((dtypes.fp8e4m3, 96.0), (dtypes.fp8e5m2, 1.25), (dtypes.half, 256.0),
             (dtypes.half, 1+2**-9), (dtypes.bfloat16, 1+2**-7))
    for dt, value in cases:
      for size, tc in ((1, 1), (32, 0)):
        with self.subTest(dtype=dt, size=size, tc=tc), Context(TC=tc):
          a = Tensor([[value]*size for _ in range(size)], dtype=dt).realize()
          result = a.matmul(a, dtype=dtypes.float32)
          self.assertEqual(result.tolist(), [[size*value*value]*size for _ in range(size)])

  @Context(TC=0)
  def test_default_dtype(self):
    for dt, v in ((dtypes.fp8e4m3, 1.25), (dtypes.fp8e5m2, 1.25), (dtypes.half, 1+2**-10), (dtypes.bfloat16, 1+2**-7)):
      with self.subTest(dtype=dt):
        a, b = Tensor([[v, v]], dtype=dt).realize(), Tensor([[v], [-1]], dtype=dt).realize()
        result = a.matmul(b)
        self.assertEqual(result.dtype, dt)
        self.assertEqual(result.realize().float().item(), v*(v-1))

  @unittest.skipUnless(dtypes.double in Device[Device.DEFAULT].renderer.supported_dtypes(), "requires float64")
  @Context(DEFAULT_FLOAT="float64")
  def test_weak_float64(self):
    for dt, a, b, expected in ((None, 1.0+2**-30, 1.0, 1.0+2**-30), (dtypes.float32, 1.0+2**-24, 3.0, 3.0+2**-22)):
      with self.subTest(dtype=dt):
        self.assertEqual(Tensor(a).reshape(1, 1).matmul(Tensor(b).reshape(1, 1), dtype=dt).tolist(), [[expected]])


if __name__ == "__main__": unittest.main()
