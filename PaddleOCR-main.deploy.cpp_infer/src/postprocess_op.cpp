// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <include/postprocess_op.h>

namespace PaddleOCR {

	void DBPostProcessor::GetContourArea(const std::vector<std::vector<float>>& box,
		float unclip_ratio, float& distance) {
		int pts_num = 4;
		float area = 0.0f;
		float dist = 0.0f;
		for (int i = 0; i < pts_num; i++) {
			area += box[i][0] * box[(i + 1) % pts_num][1] -
				box[i][1] * box[(i + 1) % pts_num][0];
			dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
				(box[i][0] - box[(i + 1) % pts_num][0]) +
				(box[i][1] - box[(i + 1) % pts_num][1]) *
				(box[i][1] - box[(i + 1) % pts_num][1]));
		}
		area = fabs(float(area / 2.0));

		distance = area * unclip_ratio / dist;
	}

	void DBPostProcessor::GetContourAreaPoint(const std::vector<cv::Point>& box,
		float unclip_ratio, float& distance) {
		int pts_num = box.size(); //4;
		float area = 0.0f;
		float dist = 0.0f;
		for (int i = 0; i < pts_num; i++) {
			area += box[i].x * box[(i + 1) % pts_num].y -
				box[i].y * box[(i + 1) % pts_num].x;
			dist += sqrtf((box[i].x - box[(i + 1) % pts_num].x) *
				(box[i].x - box[(i + 1) % pts_num].x) +
				(box[i].y - box[(i + 1) % pts_num].y) *
				(box[i].y - box[(i + 1) % pts_num].y));
		}
		area = fabs(float(area / 2.0));

		distance = area * unclip_ratio / dist;
		//std::cout << "box[0]:" << box[0].x << "," << box[0].y << std::endl;
		//std::cout << "distance:" << distance << "," << area << "," << unclip_ratio << "," << dist << std::endl;
	}

	cv::RotatedRect DBPostProcessor::UnClip(std::vector<std::vector<float>> box,
		const float& unclip_ratio) {
		float distance = 1.0;

		GetContourArea(box, unclip_ratio, distance);

		ClipperLib::ClipperOffset offset;
		ClipperLib::Path p;
		p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
			<< ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
			<< ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
			<< ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
		offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

		ClipperLib::Paths soln;
		offset.Execute(soln, distance);
		std::vector<cv::Point2f> points;

		for (int j = 0; j < soln.size(); j++) {
			for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
				points.emplace_back(soln[j][i].X, soln[j][i].Y);
			}
		}
		cv::RotatedRect res;
		if (points.size() <= 0) {
			res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
		}
		else {
			res = cv::minAreaRect(points);
		}
		return res;
	}

	cv::RotatedRect DBPostProcessor::UnClipPoint(std::vector<cv::Point> box, const float& unclip_ratio, std::vector<cv::Point> & boxes) {

		float distance = 1.0;

		GetContourAreaPoint(box, unclip_ratio, distance);

		ClipperLib::ClipperOffset offset;
		ClipperLib::Path p;
		
		for (int _i = 0; _i < box.size(); _i++) {
			p.push_back(ClipperLib::IntPoint(int(box[_i].x), int(box[_i].y)));
		}
		/*p << ClipperLib::IntPoint(int(box[0].x), int(box[0].y))
			<< ClipperLib::IntPoint(int(box[1].x), int(box[1].y))
			<< ClipperLib::IntPoint(int(box[2].x), int(box[2].y))
			<< ClipperLib::IntPoint(int(box[3].x), int(box[3].y));*/
		offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

		ClipperLib::Paths soln;
		offset.Execute(soln, distance);
		std::vector<cv::Point2f> points;

		for (int j = 0; j < soln.size(); j++) {
			for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
				points.emplace_back(soln[j][i].X, soln[j][i].Y);
			}
		}		

		cv::RotatedRect res;
		if (points.size() <= 0) {
			res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
		}
		else {
			for (int _i = 0; _i < points.size(); _i++) {
				boxes.push_back(points[_i]);
			}
			res = cv::minAreaRect(points);
		}
		return res;
	}

	float** DBPostProcessor::Mat2Vec(cv::Mat mat) {
		auto** array = new float* [mat.rows];
		for (int i = 0; i < mat.rows; ++i)
			array[i] = new float[mat.cols];
		for (int i = 0; i < mat.rows; ++i) {
			for (int j = 0; j < mat.cols; ++j) {
				array[i][j] = mat.at<float>(i, j);
			}
		}

		return array;
	}

	std::vector<std::vector<int>>
		DBPostProcessor::OrderPointsClockwise(std::vector<std::vector<int>> pts) {
		std::vector<std::vector<int>> box = pts;
		std::sort(box.begin(), box.end(), XsortInt);

		std::vector<std::vector<int>> leftmost = { box[0], box[1] };
		std::vector<std::vector<int>> rightmost = { box[2], box[3] };

		if (leftmost[0][1] > leftmost[1][1])
			std::swap(leftmost[0], leftmost[1]);

		if (rightmost[0][1] > rightmost[1][1])
			std::swap(rightmost[0], rightmost[1]);

		std::vector<std::vector<int>> rect = { leftmost[0], rightmost[0], rightmost[1],
											  leftmost[1] };
		return rect;
	}

	std::vector<std::vector<float>> DBPostProcessor::Mat2Vector(cv::Mat mat) {
		std::vector<std::vector<float>> img_vec;
		std::vector<float> tmp;

		for (int i = 0; i < mat.rows; ++i) {
			tmp.clear();
			for (int j = 0; j < mat.cols; ++j) {
				tmp.push_back(mat.at<float>(i, j));
			}
			img_vec.push_back(tmp);
		}
		return img_vec;
	}

	bool DBPostProcessor::XsortFp32(std::vector<float> a, std::vector<float> b) {
		if (a[0] != b[0])
			return a[0] < b[0];
		return false;
	}

	bool DBPostProcessor::XsortInt(std::vector<int> a, std::vector<int> b) {
		if (a[0] != b[0])
			return a[0] < b[0];
		return false;
	}

	std::vector<std::vector<float>>
		DBPostProcessor::GetMiniBoxes(cv::RotatedRect box, float& ssid) {
		ssid = std::max(box.size.width, box.size.height);

		cv::Mat points;
		cv::boxPoints(box, points);

		auto array = Mat2Vector(points);
		std::sort(array.begin(), array.end(), XsortFp32);

		std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
			idx4 = array[3];
		if (array[3][1] <= array[2][1]) {
			idx2 = array[3];
			idx3 = array[2];
		}
		else {
			idx2 = array[2];
			idx3 = array[3];
		}
		if (array[1][1] <= array[0][1]) {
			idx1 = array[1];
			idx4 = array[0];
		}
		else {
			idx1 = array[0];
			idx4 = array[1];
		}

		array[0] = idx1;
		array[1] = idx2;
		array[2] = idx3;
		array[3] = idx4;

		return array;
	}

	float DBPostProcessor::PolygonScoreAcc(std::vector<cv::Point> contour,
		cv::Mat pred) {
		int width = pred.cols;
		int height = pred.rows;
		std::vector<float> box_x;
		std::vector<float> box_y;
		for (int i = 0; i < contour.size(); ++i) {
			box_x.push_back(contour[i].x);
			box_y.push_back(contour[i].y);
		}

		int xmin =
			clamp(int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0,
				width - 1);
		int xmax =
			clamp(int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0,
				width - 1);
		int ymin =
			clamp(int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0,
				height - 1);
		int ymax =
			clamp(int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0,
				height - 1);

		cv::Mat mask;
		mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

		cv::Point* rook_point = new cv::Point[contour.size()];

		for (int i = 0; i < contour.size(); ++i) {
			rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
		}
		const cv::Point* ppt[1] = { rook_point };
		int npt[] = { int(contour.size()) };

		cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

		cv::Mat croppedImg;
		pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
			.copyTo(croppedImg);
		float score = cv::mean(croppedImg, mask)[0];

		delete[] rook_point;
		return score;
	}

	float DBPostProcessor::BoxScoreFast(std::vector<std::vector<float>> box_array,
		cv::Mat pred) {
		auto array = box_array;
		int width = pred.cols;
		int height = pred.rows;

		float box_x[4] = { array[0][0], array[1][0], array[2][0], array[3][0] };
		float box_y[4] = { array[0][1], array[1][1], array[2][1], array[3][1] };

		int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0,
			width - 1);
		int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0,
			width - 1);
		int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0,
			height - 1);
		int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0,
			height - 1);

		cv::Mat mask;
		mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

		cv::Point root_point[4];
		root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
		root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
		root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
		root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
		const cv::Point* ppt[1] = { root_point };
		int npt[] = { 4 };
		cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

		cv::Mat croppedImg;
		pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
			.copyTo(croppedImg);

		auto score = cv::mean(croppedImg, mask)[0];
		return score;
	}

	float DBPostProcessor::BoxScoreFastPoint(std::vector<cv::Point> box_array,
		cv::Mat pred) {
		auto array = box_array;
		int width = pred.cols;
		int height = pred.rows;

		int arr_size = array.size();
		float* box_x = new float[arr_size];
		float* box_y = new float[arr_size];
		for (int i = 0; i < arr_size; i++) {
			box_x[i] = array[i].x;
			box_y[i] = array[i].y;
		}

		int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + arr_size)))), 0,
			width - 1);
		int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + arr_size)))), 0,
			width - 1);
		int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + arr_size)))), 0,
			height - 1);
		int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + arr_size)))), 0,
			height - 1);

		cv::Mat mask;
		mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

		cv::Point* root_point = new cv::Point[arr_size];
		for (int i = 0; i < arr_size; i++) {
			root_point[i] = cv::Point(int(array[i].x) - xmin, int(array[i].y) - ymin);
		}
		/*root_point[0] = cv::Point(int(array[0].x) - xmin, int(array[0].y) - ymin);
		root_point[1] = cv::Point(int(array[1].x) - xmin, int(array[1].y) - ymin);
		root_point[2] = cv::Point(int(array[2].x) - xmin, int(array[2].y) - ymin);
		root_point[3] = cv::Point(int(array[3].x) - xmin, int(array[3].y) - ymin);*/
		const cv::Point* ppt[1] = { root_point };
		int npt[] = { arr_size };
		cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

		/*std::cout << root_point[0].x << "," << root_point[0].y << " "
			<< root_point[1].x << "," << root_point[1].y << " "
			<< root_point[2].x << "," << root_point[2].y << " "
			<< root_point[3].x << "," << root_point[3].y << " " << std::endl;*/

		cv::Mat croppedImg;
		pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
			.copyTo(croppedImg);

		//std::cout << xmin << "," << xmax << " " << ymin << "," << ymax << " " << std::endl;

		auto score = cv::mean(croppedImg, mask)[0];
		return score;
	}

	std::vector<std::vector<std::vector<int>>> DBPostProcessor::BoxesFromBitmap(
		const cv::Mat pred, const cv::Mat bitmap, const float& box_thresh,
		const float& det_db_unclip_ratio, const std::string& det_db_score_mode) {
		const int min_size = 3;
		const int max_candidates = 1000;

		int width = bitmap.cols;
		int height = bitmap.rows;

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
			cv::CHAIN_APPROX_SIMPLE);

		int num_contours =
			contours.size() >= max_candidates ? max_candidates : contours.size();

		std::vector<std::vector<std::vector<int>>> boxes;

		for (int _i = 0; _i < num_contours; _i++) {
			if (contours[_i].size() <= 2) {
				continue;
			}
			float ssid;
			cv::RotatedRect box = cv::minAreaRect(contours[_i]);
			auto array = GetMiniBoxes(box, ssid);

			auto box_for_unclip = array;
			// end get_mini_box

			if (ssid < min_size) {
				continue;
			}

			float score;
			if (det_db_score_mode == "slow")
				/* compute using polygon*/
				score = PolygonScoreAcc(contours[_i], pred);
			else
				score = BoxScoreFast(array, pred);

			if (score < box_thresh)
				continue;

			// start for unclip
			cv::RotatedRect points = UnClip(box_for_unclip, det_db_unclip_ratio);
			if (points.size.height < 1.001 && points.size.width < 1.001) {
				continue;
			}
			// end for unclip

			cv::RotatedRect clipbox = points;
			auto cliparray = GetMiniBoxes(clipbox, ssid);

			if (ssid < min_size + 2)
				continue;

			int dest_width = pred.cols;
			int dest_height = pred.rows;
			std::vector<std::vector<int>> intcliparray;

			for (int num_pt = 0; num_pt < 4; num_pt++) {
				std::vector<int> a{ int(clampf(roundf(cliparray[num_pt][0] / float(width) *
													 float(dest_width)),
											  0, float(dest_width))),
								   int(clampf(roundf(cliparray[num_pt][1] /
													 float(height) * float(dest_height)),
											  0, float(dest_height))) };
				intcliparray.push_back(a);
			}
			boxes.push_back(intcliparray);

		} // end for
		return boxes;
	}


	std::vector<std::vector<std::vector<int>>> DBPostProcessor::PolygonsFromBitmap(
		const cv::Mat pred, const cv::Mat bitmap, const float& box_thresh,
		const float& det_db_unclip_ratio, const std::string& det_db_score_mode,
		const float& src_width,const float& src_height) {

		std::vector<std::vector<std::vector<int>>> boxes;

		const int min_size = 3;
		//const int max_candidates = 1000;

		int width = bitmap.cols;
		int height = bitmap.rows;


		int dest_width = src_width; // pred.cols;
		int dest_height = src_height;// pred.rows;
		//std::cout << width << "," << height << ":" << dest_width << "," << dest_height << std::endl;

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
			cv::CHAIN_APPROX_SIMPLE);

		/*for (int i = 0; i < contours.size(); i++) {
			std::cout << i << ":  ";
			for (int j = 0; j < contours[i].size(); j++) {
				std::cout << contours[i][j].x << "," << contours[i][j].y << " ";
			}
			std::cout << std::endl;
		}*/

		for (int _c = 0; _c < contours.size(); _c++) {
			float ssid;
			auto& contour = contours[_c];
			auto epsilon = 0.002 * cv::arcLength(contour, true);
			std::vector<cv::Point> approx;
			cv::approxPolyDP(contour, approx, epsilon, true);

			/*for (int i = 0; i < approx.size(); i++) {
					std::cout << approx[i].x << "," << approx[i].y << " ";
			}
			std::cout << std::endl;*/

			if (approx.size() < 4) {
				continue;
			}

			auto score = BoxScoreFastPoint(approx, pred);
			//std::cout << "============score" << _c << ":==" << score << "==" << box_thresh << std::endl;
			if (score < box_thresh)
				continue;


			if (approx.size() <= 2) {
				continue;
			}
			
			std::vector<cv::Point> point_boxes;

			cv::RotatedRect points = UnClipPoint(approx, det_db_unclip_ratio, point_boxes);
			if (points.size.height < 1.001 && points.size.width < 1.001) {
				continue;
			}

			cv::RotatedRect clipbox = points;
			auto cliparray = GetMiniBoxes(clipbox, ssid);

			if (ssid < min_size + 2)
				continue;

			//std::cout << "points" << point_boxes.size() << std::endl;

			/*cv::Mat boxMat;
			cv::boxPoints(points, boxMat);

			auto array = Mat2Vector(boxMat);*/

								


			std::vector<std::vector<int>> intcliparray;

			/*for (int num_pt = 0; num_pt < cliparray.size(); num_pt++) {
				std::vector<int> a{ int(clampf(roundf(cliparray[num_pt][0] / float(width) *
														float(dest_width)),
												0, float(dest_width))),
									int(clampf(roundf(cliparray[num_pt][1] /
														float(height) * float(dest_height)),
												0, float(dest_height))) };
				intcliparray.push_back(a);
			}*/
			for (int num_pt = 0; num_pt < point_boxes.size(); num_pt++) {
				//std::cout << point_boxes[num_pt].x << "," << point_boxes[num_pt].y << std::endl;
				std::vector<int> a{ int(clampf(roundf(point_boxes[num_pt].x / float(width) *
														float(dest_width)),
												0, float(dest_width))),
									int(clampf(roundf(point_boxes[num_pt].y /
														float(height) * float(dest_height)),
												0, float(dest_height))) };
				intcliparray.push_back(a);
				//std::cout << a[0] << "," << a[1] << std::endl;
			}
			boxes.push_back(intcliparray);
		}
		/*for (int i = 0; i < boxes.size(); i++) {
			PaddleOCR::Utility::Print_vec_int(boxes[i], "PolygonsFromBitmap" + std::to_string(i));
		}*/
		//std::cout << "PolygonsFromBitmap End" << std::endl;
		return boxes;
	}

	std::vector<std::vector<std::vector<int>>> DBPostProcessor::FilterTagDetRes(
		std::vector<std::vector<std::vector<int>>> boxes, float ratio_h,
		float ratio_w, cv::Mat srcimg) {
		int oriimg_h = srcimg.rows;
		int oriimg_w = srcimg.cols;

		std::vector<std::vector<std::vector<int>>> root_points;
		for (int n = 0; n < boxes.size(); n++) {
			boxes[n] = OrderPointsClockwise(boxes[n]);
			for (int m = 0; m < boxes[0].size(); m++) {
				boxes[n][m][0] /= ratio_w;
				boxes[n][m][1] /= ratio_h;

				boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
				boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
			}
		}

		for (int n = 0; n < boxes.size(); n++) {
			int rect_width, rect_height;
			rect_width = int(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
				pow(boxes[n][0][1] - boxes[n][1][1], 2)));
			rect_height = int(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
				pow(boxes[n][0][1] - boxes[n][3][1], 2)));
			if (rect_width <= 4 || rect_height <= 4)
				continue;
			root_points.push_back(boxes[n]);
		}
		return root_points;
	}

	std::vector<std::vector<std::vector<int>>> DBPostProcessor::FilterTagDetResOnlyClip(
		std::vector<std::vector<std::vector<int>>> boxes, float ratio_h,
		float ratio_w, cv::Mat srcimg) {
		int oriimg_h = srcimg.rows;
		int oriimg_w = srcimg.cols;
		//std::cout << "srcimg" << oriimg_w << ":" << oriimg_h << std::endl;
		//std::cout << "ratio" << ratio_w << ":" << ratio_h << std::endl;
		std::vector<std::vector<std::vector<int>>> root_points;
		for (int n = 0; n < boxes.size(); n++) {
			//boxes[n] = OrderPointsClockwise(boxes[n]);
			for (int m = 0; m < boxes[n].size(); m++) {
				//boxes[n][m][0] /= ratio_w;
				//boxes[n][m][1] /= ratio_h;

				boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
				boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
			}
		}
		int max_points = 0;
		for (int n = 0; n < boxes.size(); n++) {
			/*int rect_width, rect_height;
			rect_width = int(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
				pow(boxes[n][0][1] - boxes[n][1][1], 2)));
			rect_height = int(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
				pow(boxes[n][0][1] - boxes[n][3][1], 2)));
			if (rect_width <= 4 || rect_height <= 4)
				continue;*/
			root_points.push_back(boxes[n]);
			max_points = _max(max_points, boxes[n].size());
		}
		//std::cout << "max_points" << max_points << std::endl;
		if (max_points > 0) {
			for (int n = 0; n < root_points.size(); n++) {
				auto& b = root_points[n];
				int padding_size = max_points - b.size();
				//std::cout << "padding_size" << padding_size << std::endl;
				if (padding_size == 0) {
					continue;
				}
				auto last_point = b.back();

				for (int i = 0; i < padding_size; i++) {
					b.push_back(last_point);
				}
				/*std::vector<std::vector<int>> padding;
				for (int i = 0; i < padding_size; i++) {
					padding.push_back(last_point);
				}*/

			}
		}

		//std::cout << "size" << root_points.size() << std::endl;
		return root_points;
	}

	void TablePostProcessor::init(std::string label_path,
		bool merge_no_span_structure) {
		this->label_list_ = Utility::ReadDict(label_path);
		if (merge_no_span_structure) {
			this->label_list_.push_back("<td></td>");
			std::vector<std::string>::iterator it;
			for (it = this->label_list_.begin(); it != this->label_list_.end();) {
				if (*it == "<td>") {
					it = this->label_list_.erase(it);
				}
				else {
					++it;
				}
			}
		}
		// add_special_char
		this->label_list_.insert(this->label_list_.begin(), this->beg);
		this->label_list_.push_back(this->end);
	}

	void TablePostProcessor::Run(
		std::vector<float>& loc_preds, std::vector<float>& structure_probs,
		std::vector<float>& rec_scores, std::vector<int>& loc_preds_shape,
		std::vector<int>& structure_probs_shape,
		std::vector<std::vector<std::string>>& rec_html_tag_batch,
		std::vector<std::vector<std::vector<int>>>& rec_boxes_batch,
		std::vector<int>& width_list, std::vector<int>& height_list) {
		for (int batch_idx = 0; batch_idx < structure_probs_shape[0]; batch_idx++) {
			// image tags and boxs
			std::vector<std::string> rec_html_tags;
			std::vector<std::vector<int>> rec_boxes;

			float score = 0.f;
			int count = 0;
			float char_score = 0.f;
			int char_idx = 0;

			// step
			for (int step_idx = 0; step_idx < structure_probs_shape[1]; step_idx++) {
				std::string html_tag;
				std::vector<int> rec_box;
				// html tag
				int step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) *
					structure_probs_shape[2];
				char_idx = int(Utility::argmax(
					&structure_probs[step_start_idx],
					&structure_probs[step_start_idx + structure_probs_shape[2]]));
				char_score = float(*std::max_element(
					&structure_probs[step_start_idx],
					&structure_probs[step_start_idx + structure_probs_shape[2]]));
				html_tag = this->label_list_[char_idx];

				if (step_idx > 0 && html_tag == this->end) {
					break;
				}
				if (html_tag == this->beg) {
					continue;
				}
				count += 1;
				score += char_score;
				rec_html_tags.push_back(html_tag);

				// box
				if (html_tag == "<td>" || html_tag == "<td" || html_tag == "<td></td>") {
					for (int point_idx = 0; point_idx < loc_preds_shape[2]; point_idx++) {
						step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) *
							loc_preds_shape[2] +
							point_idx;
						float point = loc_preds[step_start_idx];
						if (point_idx % 2 == 0) {
							point = int(point * width_list[batch_idx]);
						}
						else {
							point = int(point * height_list[batch_idx]);
						}
						rec_box.push_back(point);
					}
					rec_boxes.push_back(rec_box);
				}
			}
			score /= count;
			if (std::isnan(score) || rec_boxes.size() == 0) {
				score = -1;
			}
			rec_scores.push_back(score);
			rec_boxes_batch.push_back(rec_boxes);
			rec_html_tag_batch.push_back(rec_html_tags);
		}
	}

	void PicodetPostProcessor::init(std::string label_path,
		const double score_threshold,
		const double nms_threshold,
		const std::vector<int>& fpn_stride) {
		this->label_list_ = Utility::ReadDict(label_path);
		this->score_threshold_ = score_threshold;
		this->nms_threshold_ = nms_threshold;
		this->num_class_ = label_list_.size();
		this->fpn_stride_ = fpn_stride;
	}

	void PicodetPostProcessor::Run(std::vector<StructurePredictResult>& results,
		std::vector<std::vector<float>> outs,
		std::vector<int> ori_shape,
		std::vector<int> resize_shape, int reg_max) {
		int in_h = resize_shape[0];
		int in_w = resize_shape[1];
		float scale_factor_h = resize_shape[0] / float(ori_shape[0]);
		float scale_factor_w = resize_shape[1] / float(ori_shape[1]);

		std::vector<std::vector<StructurePredictResult>> bbox_results;
		bbox_results.resize(this->num_class_);
		for (int i = 0; i < this->fpn_stride_.size(); ++i) {
			int feature_h = std::ceil((float)in_h / this->fpn_stride_[i]);
			int feature_w = std::ceil((float)in_w / this->fpn_stride_[i]);
			for (int idx = 0; idx < feature_h * feature_w; idx++) {
				// score and label
				float score = 0;
				int cur_label = 0;
				for (int label = 0; label < this->num_class_; label++) {
					if (outs[i][idx * this->num_class_ + label] > score) {
						score = outs[i][idx * this->num_class_ + label];
						cur_label = label;
					}
				}
				// bbox
				if (score > this->score_threshold_) {
					int row = idx / feature_w;
					int col = idx % feature_w;
					std::vector<float> bbox_pred(
						outs[i + this->fpn_stride_.size()].begin() + idx * 4 * reg_max,
						outs[i + this->fpn_stride_.size()].begin() +
						(idx + 1) * 4 * reg_max);
					bbox_results[cur_label].push_back(
						this->disPred2Bbox(bbox_pred, cur_label, score, col, row,
							this->fpn_stride_[i], resize_shape, reg_max));
				}
			}
		}
		for (int i = 0; i < bbox_results.size(); i++) {
			bool flag = bbox_results[i].size() <= 0;
		}
		for (int i = 0; i < bbox_results.size(); i++) {
			bool flag = bbox_results[i].size() <= 0;
			if (bbox_results[i].size() <= 0) {
				continue;
			}
			this->nms(bbox_results[i], this->nms_threshold_);
			for (auto box : bbox_results[i]) {
				box.box[0] = box.box[0] / scale_factor_w;
				box.box[2] = box.box[2] / scale_factor_w;
				box.box[1] = box.box[1] / scale_factor_h;
				box.box[3] = box.box[3] / scale_factor_h;
				results.push_back(box);
			}
		}
	}

	StructurePredictResult
		PicodetPostProcessor::disPred2Bbox(std::vector<float> bbox_pred, int label,
			float score, int x, int y, int stride,
			std::vector<int> im_shape, int reg_max) {
		float ct_x = (x + 0.5) * stride;
		float ct_y = (y + 0.5) * stride;
		std::vector<float> dis_pred;
		dis_pred.resize(4);
		for (int i = 0; i < 4; i++) {
			float dis = 0;
			std::vector<float> bbox_pred_i(bbox_pred.begin() + i * reg_max,
				bbox_pred.begin() + (i + 1) * reg_max);
			std::vector<float> dis_after_sm =
				Utility::activation_function_softmax(bbox_pred_i);
			for (int j = 0; j < reg_max; j++) {
				dis += j * dis_after_sm[j];
			}
			dis *= stride;
			dis_pred[i] = dis;
		}

		float xmin = (std::max)(ct_x - dis_pred[0], .0f);
		float ymin = (std::max)(ct_y - dis_pred[1], .0f);
		float xmax = (std::min)(ct_x + dis_pred[2], (float)im_shape[1]);
		float ymax = (std::min)(ct_y + dis_pred[3], (float)im_shape[0]);

		StructurePredictResult result_item;
		result_item.box = { xmin, ymin, xmax, ymax };
		result_item.type = this->label_list_[label];
		result_item.confidence = score;

		return result_item;
	}

	void PicodetPostProcessor::nms(std::vector<StructurePredictResult>& input_boxes,
		float nms_threshold) {
		std::sort(input_boxes.begin(), input_boxes.end(),
			[](StructurePredictResult a, StructurePredictResult b) {
				return a.confidence > b.confidence;
			});
		std::vector<int> picked(input_boxes.size(), 1);

		for (int i = 0; i < input_boxes.size(); ++i) {
			if (picked[i] == 0) {
				continue;
			}
			for (int j = i + 1; j < input_boxes.size(); ++j) {
				if (picked[j] == 0) {
					continue;
				}
				float iou = Utility::iou(input_boxes[i].box, input_boxes[j].box);
				if (iou > nms_threshold) {
					picked[j] = 0;
				}
			}
		}
		std::vector<StructurePredictResult> input_boxes_nms;
		for (int i = 0; i < input_boxes.size(); ++i) {
			if (picked[i] == 1) {
				input_boxes_nms.push_back(input_boxes[i]);
			}
		}
		input_boxes = input_boxes_nms;
	}

} // namespace PaddleOCR
