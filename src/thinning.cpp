/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "thinning.h"

using namespace cv;

// Applies a thinning iteration to a binary image
static void thinningIteration(Mat img, int iter, int thinningType)
{
	Mat marker = Mat::zeros(img.size(), CV_8UC1);

	if(thinningType == THINNING_ZHANGSUEN)
	{
		for (int i = 1; i < img.rows-1; i++)
		{
			for (int j = 1; j < img.cols-1; j++)
			{
				uchar p2 = img.at<uchar>(i-1, j);
				uchar p3 = img.at<uchar>(i-1, j+1);
				uchar p4 = img.at<uchar>(i, j+1);
				uchar p5 = img.at<uchar>(i+1, j+1);
				uchar p6 = img.at<uchar>(i+1, j);
				uchar p7 = img.at<uchar>(i+1, j-1);
				uchar p8 = img.at<uchar>(i, j-1);
				uchar p9 = img.at<uchar>(i-1, j-1);

				int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
							(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
							(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
							(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
				int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
				int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
				int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

				if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
					marker.at<uchar>(i,j) = 1;
			}
		}
	}

	if(thinningType == THINNING_GUOHALL)
	{
		for (int i = 1; i < img.rows-1; i++)
		{
			for (int j = 1; j < img.cols-1; j++)
			{
				uchar p2 = img.at<uchar>(i-1, j);
				uchar p3 = img.at<uchar>(i-1, j+1);
				uchar p4 = img.at<uchar>(i, j+1);
				uchar p5 = img.at<uchar>(i+1, j+1);
				uchar p6 = img.at<uchar>(i+1, j);
				uchar p7 = img.at<uchar>(i+1, j-1);
				uchar p8 = img.at<uchar>(i, j-1);
				uchar p9 = img.at<uchar>(i-1, j-1);

				int C  = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) +
							((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
				int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
				int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
				int N  = N1 < N2 ? N1 : N2;
				int m  = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);

				if ((C == 1) && ((N >= 2) && ((N <= 3)) & (m == 0)))
					marker.at<uchar>(i,j) = 1;
			}
		}
	}

	img &= ~marker;
}

// Apply the thinning procedure to a given image
void thinning(InputArray input, OutputArray output, int thinningType)
{
	Mat processed = input.getMat().clone();
	CV_CheckTypeEQ(processed.type(), CV_8UC1, "");
	// Enforce the range of the input image to be in between 0 - 255
	processed /= 255;

	Mat prev = Mat::zeros(processed.size(), CV_8UC1);
	Mat diff;

	do
	{
		thinningIteration(processed, 0, thinningType);
		thinningIteration(processed, 1, thinningType);
		absdiff(processed, prev, diff);
		processed.copyTo(prev);
	}
	while (countNonZero(diff) > 0);

	processed *= 255;

	output.assign(processed);
}
