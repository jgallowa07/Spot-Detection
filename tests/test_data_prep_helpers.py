
from __future__ import print_function
from __future__ import division

import sys
#sys.path.insert(0, ' ../scripts/')

import numpy as np

#from helpers import *
import tests
from DataPrep.helpers import *
import numpy as np

np.random.seed(23)

class TestHelpers(tests.testDataPrep):
    """
    Tests for the TsEncoder class.
    """

    def test_dot_click_annoation_file_to_pixelmap(self):
        """
        Let's test that the correct dimentions are created for each new layer
        """
        # note, filepaths are relative to where you run nose.
        # TODO prep example in super class, see tsenc
        width, height = 1024, 1024
        anno_file = "./Data/Annotation/annotation_output/L1-D01-g_output.csv"

        # make sure the images are the same size        
        pixelmap = dot_click_annoation_file_to_pixelmap(
            anno_file = anno_file,
            width = width,
            height = height,
            dot_radius = 2)
        self.assertEqual(pixelmap.shape,(width,height))

        # assert that the number of 1's is geater than number of clicks
        anno_line_count = len(open(anno_file).readlines(  )) 
        self.assertTrue(np.sum(pixelmap) >= anno_line_count)
    
        # TODO more functions.

    
    def test_symquant_to_pixelmap(self):
        """
        Tests for symquant -> pixelmap
        """
        pass

    def test_colocalization(self):
        """
        tests for colocalization.
        """
        pass





