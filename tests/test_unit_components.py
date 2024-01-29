from MCQA_Benchmark.common.utils import *
from decorator import decorator 

def test_retry():
    @retry_on_failure(3,1)
    def test_function():
        raise Exception("Test exception")
    
    assert test_function() == 1

test_retry()