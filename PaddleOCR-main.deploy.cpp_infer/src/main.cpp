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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/shape/shape_transformer.hpp"
#include <iostream>
#include <vector>

#include <include/args.h>
#include <include/paddleocr.h>
#include <include/paddlestructure.h>

#define PI acos(-1)

constexpr auto is_debug = false;

using namespace PaddleOCR;

PPOCR ocr;

char* get_result_json(const std::vector<OCRPredictResult>& ocr_results)
{
    std::stringstream result;
    std::vector<OCRPredictResult> filter_ocr;

    for (const auto& r : ocr_results)
    {
        if (r.score > 0)
        {
            filter_ocr.push_back(r);
        }
    }

    result << "[";

    for (int i = 0; i < filter_ocr.size(); i++)
    {
        result << "{";
        result << R"("text":")" << filter_ocr[i].text << "\",";
        result << "\"score\":" << filter_ocr[i].score;
        result << "}";
        if (i != filter_ocr.size() - 1)
        {
            result << ',';
        }
    }

    result << "]";

    return _strdup(result.str().c_str());
}

template <typename T = double>
void print_points(const std::vector<std::vector<T>>& points)
{
    for (auto p : points)
    {
        std::cout << "[" << p[0] << "," << p[1] << "], ";
    }
    std::cout << std::endl;
}

void draw_points(const cv::Mat& img, const std::vector<std::vector<double>>& points, const cv::Scalar& color,
                 const bool is_text = false)
{
    for (int i = 0; i < points.size(); ++i)
    {
        const auto p = cv::Point(int(points[i][0]), int(points[i][1]));
        if (is_text)
        {
            cv::putText(img, std::to_string(i), p, cv::FONT_HERSHEY_COMPLEX, 1, color, 1, 4);
        }
        cv::circle(img, p, 2, color, 3, cv::LINE_AA);
    }
}

template <typename T1, typename T2>
std::vector<std::vector<T2>> convert_vec_type(const std::vector<std::vector<T1>>& points)
{
    std::vector<std::vector<T2>> result;
    for (auto p : points)
    {
        result.push_back({T2(p[0]), T2(p[1])});
    }
    return result;
}


std::vector<std::vector<std::vector<int>>> det(const cv::Mat& img)
{
    std::vector<std::vector<std::vector<int>>> det_boxes;
    ocr.keke_det(img, det_boxes);
    return det_boxes;
}

std::vector<OCRPredictResult> rec(const std::vector<cv::Mat>& img_list)
{
    return ocr.keke_rec(img_list);
}


double points_len(const std::vector<double>& point1, const std::vector<double>& point2)
{
    return sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2));
}


double get_in_circle(const std::vector<std::vector<double>>& points, const std::vector<double>& center)
{
    double radius = 10000;
    for (const auto& p : points)
    {
        const auto r = points_len(p, center);
        if (r < radius)
        {
            radius = r;
        }
    }
    return radius;
}


std::vector<std::vector<double>> split_arc_into_segments(const std::vector<double>& point_a,
                                                         const std::vector<double>& point_b,
                                                         const std::vector<double>& center, const double radius,
                                                         const int num_segments = 5)
{
    const double vec_a[] = {point_a[0] - center[0], point_a[1] - center[1]};
    const double vec_b[] = {point_b[0] - center[0], point_b[1] - center[1]};
    const auto angle_a = atan2(vec_a[1], vec_a[0]);
    auto angle_b = atan2(vec_b[1], vec_b[0]);
    if (angle_b < angle_a)
    {
        angle_b += 2 * PI;
    }
    const auto angle_difference = angle_b - angle_a;
    const auto segment_angle = angle_difference / (num_segments - 1);
    std::vector<std::vector<double>> points;
    for (int i = 0; i < num_segments; ++i)
    {
        const auto ang = angle_a + i * segment_angle;
        const auto x = center[0] + radius * cos(ang);
        const auto y = center[1] + radius * sin(ang);
        points.push_back({x, y});
    }
    return points;
}


double calculate_arc_length(const std::vector<double>& p1, const std::vector<double>& p2,
                            const std::vector<double>& center,
                            const double radius)
{
    // std::cout << "calculate_arc_length: start" << std::endl;
    const double vec_oa[] = {p1[0] - center[0], p1[1] - center[1]};
    // std::cout << "calculate_arc_length: step1" << std::endl;
    // std::cout << "calculate_arc_length: step2 " << p2[0] - center[0] << "," << p2[1] - center[1] << std::endl;
    const double vec_ob[] = {p2[0] - center[0], p2[1] - center[1]};
    const auto norm_oa = sqrt(pow(vec_oa[0], 2) + pow(vec_oa[1], 2));
    const auto norm_ob = sqrt(pow(vec_ob[0], 2) + pow(vec_ob[1], 2));
    const auto dot_product = vec_oa[0] * vec_ob[0] + vec_oa[1] * vec_ob[1];
    const auto cos_theta = dot_product / (norm_oa * norm_ob);
    const auto theta = acos(cos_theta);
    const auto arc_length = radius * theta;
    // std::cout << "calculate_arc_length: end" << std::endl;
    return arc_length;
}

double radians(const double degrees)
{
    return degrees * PI / 180.0;
}

double degrees(const double radians)
{
    return radians * 180.0 / PI;
}


double angle_with_y_axis(const std::vector<double>& p1, const std::vector<double>& p2)
{
    // std::cout << "angle_with_y_axis: center" << p1[0] << "," << p1[1] << " " << p2[0] << "," << p2[1] << std::endl;
    const double vec_p1_p2[] = {p2[0] - p1[0], p2[1] - p1[1]};
    const double vec_p1_p3[] = {0.0, -100.0};
    const auto mag_p1_p2 = sqrt(pow(vec_p1_p2[0], 2) + pow(vec_p1_p2[1], 2));
    const auto mag_p1_p3 = sqrt(pow(vec_p1_p3[0], 2) + pow(vec_p1_p3[1], 2));
    const auto dot_product = vec_p1_p2[0] * vec_p1_p3[0] + vec_p1_p2[1] * vec_p1_p3[1];
    const auto angle_rad = acos(dot_product / (mag_p1_p2 * mag_p1_p3));
    const auto cross_product = vec_p1_p2[0] * vec_p1_p3[1] - vec_p1_p2[1] * vec_p1_p3[0];
    double angle_deg;
    if (cross_product < 0)
    {
        angle_deg = degrees(angle_rad);
    }
    else
    {
        angle_deg = 360.0 - degrees(angle_rad);
    }
    return 360.0 - angle_deg;
}

std::vector<double> offset_circle_center(const std::vector<std::vector<double>>& min_box,
                                         const std::vector<double>& circle_center)
{
    // std::cout << "offset_circle_center : start" << std::endl;
    auto len1 = points_len(min_box[0], min_box[1]);
    auto len2 = points_len(min_box[0], min_box[3]);
    double k = 0;
    if (len1 > len2)
        k = (min_box[0][1] - min_box[1][1]) / (min_box[0][0] - min_box[1][0]);
    else
        k = (min_box[0][1] - min_box[3][1]) / (min_box[0][0] - min_box[3][0]);
    std::vector<double> box_center = {(min_box[2][0] + min_box[0][0]) / 2, (min_box[2][1] + min_box[0][1]) / 2};

    auto x = (pow(k, 2) * circle_center[0] + box_center[0] + k * (box_center[1] - circle_center[1])) / (pow(k, 2) + 1);
    auto y = k * (x - circle_center[0]) + circle_center[1];
    // std::cout << "offset_circle_center : end" << std::endl;
    return {x, y};
}

std::vector<double> rotate_point(const std::vector<double>& point, const std::vector<double>& center,
                                 const double angle)
{
    const auto translated_x = point[0] - center[0];
    const auto translated_y = point[1] - center[1];
    const auto r = radians(angle);
    const auto rotated_x = translated_x * cos(r) + translated_y * -sin(r);
    const auto rotated_y = translated_x * sin(r) + translated_y * cos(r);
    const auto new_p_x = rotated_x + center[0];
    const auto new_p_y = rotated_y + center[1];
    std::vector<double> new_point = {round(new_p_x), round(new_p_y)};
    return new_point;
}

cv::Mat get_min_img(const cv::Mat& img, const std::vector<cv::Point2f>& min_box, const double width,
                    const double height)
{
    const cv::Point2f pts_std[] =
    {
        {0, 0},
        {0, static_cast<float>(height)},
        {static_cast<float>(width), static_cast<float>(height)},
        {static_cast<float>(width), 0},
    };
    cv::Mat dst_img;
    const cv::Point2f boxes[] = {min_box[0], min_box[1], min_box[2], min_box[3]};
    const auto m = cv::getPerspectiveTransform(boxes, pts_std);
    cv::warpPerspective(
        img,
        dst_img,
        m,
        cv::Size(width, height),
        cv::INTER_CUBIC,
        cv::BORDER_REPLICATE
    );
    return dst_img;
}

std::vector<double> get_minarea_sorted_rect(const std::vector<std::vector<double>>& points,
                                            std::vector<std::vector<double>>& box)
{
    // std::cout << "get_minarea_sorted_rect: start" << std::endl;
    std::vector<cv::Point2f> boxes;
    for (auto& p : points)
    {
        boxes.push_back(cv::Point2f(static_cast<float>(p[0]), static_cast<float>(p[1])));
    }
    const auto bounding_box = cv::minAreaRect(boxes);
    // std::cout << "get_minarea_sorted_rect: bounding_box" << std::endl;
    cv::Mat box_points;
    cv::boxPoints(bounding_box, box_points);
    const auto mat_box = Utility::Mat2Vector(box_points);
    box = convert_vec_type<int, double>(mat_box);
    std::vector<double> center = {(box[2][0] + box[0][0]) / 2, (box[2][1] + box[0][1]) / 2};
    // std::cout << "get_minarea_sorted_rect: end" << std::endl;
    return center;
}

std::vector<std::vector<double>> expend_min_box(const std::vector<std::vector<double>>& min_box, const double padding_x,
                                                const double padding_y)
{
    std::vector<std::vector<double>> result = {
        {min_box[0][0] - padding_x, min_box[0][1] - padding_y},
        {min_box[1][0] - padding_x, min_box[1][1] + padding_y},
        {min_box[2][0] + padding_x, min_box[2][1] + padding_y},
        {min_box[3][0] + padding_x, min_box[3][1] - padding_y}
    };
    return result;
}


std::vector<std::vector<double>> min_box_sort(const std::vector<std::vector<double>>& min_box,
                                              const std::vector<double>& min_box_center)
{
    // std::cout << "min_box_sort: center" << min_box_center[0] << "," << min_box_center[1] << std::endl;
    // print_points(min_box);
    int p0 = 0;
    int p1 = 0;
    int p2 = 0;
    int p3 = 0;
    for (int i = 0; i < 4; ++i)
    {
        if (min_box[i][0] < min_box_center[0] && min_box[i][1] < min_box_center[1])
        {
            p0 = i;
        }
        else if (min_box[i][0] < min_box_center[0] && min_box[i][1] > min_box_center[1])
        {
            p1 = i;
        }
        else if (min_box[i][0] > min_box_center[0] && min_box[i][1] > min_box_center[1])
        {
            p2 = i;
        }
        else if (min_box[i][0] > min_box_center[0] && min_box[i][1] < min_box_center[1])
        {
            p3 = i;
        }
    }


    return {min_box[p0], min_box[p1], min_box[p2], min_box[p3]};
}


std::vector<double> point_on_circle(const std::vector<double>& out_point, const std::vector<double>& center,
                                    const double radius)
{
    const auto out_width = out_point[0] - center[0];
    const auto out_height = out_point[1] - center[1];
    const auto out_hypotenuse = sqrt(pow(out_width, 2) + pow(out_height, 2));
    const auto y = out_height * radius / out_hypotenuse;
    auto x = fabs(out_width * radius / out_hypotenuse);
    if (out_point[0] < center[0])
        x = -x;
    return {x + center[0], y + center[1]};
}


double point_2_circle_distance(const std::vector<double>& point, const std::vector<double>& center, const double radius)
{
    const auto x = point[0];
    const auto y = point[1];
    const auto cx = center[0];
    const auto cy = center[1];
    const auto point_len = sqrt(pow(x - cx, 2) + pow(y - cy, 2));
    const auto distance = point_len - radius;
    return distance;
}

double point_to_line(const std::vector<double>& line_point1, const std::vector<double>& line_point2,
                     const std::vector<double>& point)
{
    const std::vector<double> v = {line_point2[0] - line_point1[0], line_point2[1] - line_point1[1]};
    const std::vector<double> w = {point[0] - line_point1[0], point[1] - line_point1[1]};
    const auto c = sqrt(v[0] * v[0] + v[1] * v[1]);
    if (c == 0)
        return points_len(point, line_point1);
    const auto proj = (w[0] * v[0] + w[1] * v[1]) / pow(c, 2);
    std::vector<double> closest;
    if (proj < 0)
        closest = line_point1;
    else if (proj > 1)
        closest = line_point2;
    else
        closest = {line_point1[0] + proj * v[0], line_point1[1] + proj * v[1]};
    return points_len(point, closest);
}

std::vector<double> tu_point(const std::vector<std::vector<double>>& min_box,
                             const std::vector<std::vector<double>>& points)
{
    std::vector<double> box_center = {(min_box[2][0] + min_box[0][0]) / 2, (min_box[2][1] + min_box[0][1]) / 2};
    auto len1 = points_len(min_box[0], min_box[1]);
    auto len3 = points_len(min_box[0], min_box[3]);

    std::vector<std::vector<std::vector<double>>> width_points;
    double width = 0;
    double height = 0;

    if (len1 > len3)
    {
        width = len1;
        height = len3;
        width_points = {{min_box[0], min_box[1]}, {min_box[2], min_box[3]}};
    }

    else
    {
        width = len3;
        height = len1;
        width_points = {{min_box[0], min_box[3]}, {min_box[2], min_box[1]}};
    }
    double min_len0 = 10000;
    double min_len1 = 10000;
    std::vector<double> width_point0;
    std::vector<double> width_point1;
    for (auto p : points)
    {
        auto l0 = point_to_line(width_points[0][0], width_points[0][1], p);
        if (l0 < min_len0)
        {
            min_len0 = l0;
            width_point0 = p;
        }
        auto l1 = point_to_line(width_points[1][0], width_points[1][1], p);
        if (l1 < min_len1)
        {
            min_len1 = l1;
            width_point1 = p;
        }
    }
    auto len_p0 = points_len(width_point0, box_center);
    auto len_p1 = points_len(width_point1, box_center);
    if (len_p0 < len_p1)
        return width_point0;
    return width_point1;
}

cv::Mat tps_img(const std::vector<std::vector<double>>& points, const cv::Mat& src_image,
                std::vector<cv::Point>& tps_target)
{
    // std::cout << "tps_img: start" << std::endl;
    const auto len = points.size();
    auto y_top = points[0][1];
    auto y_bottom = points[len - 1][1];
    auto x_top = points[0][0];
    auto x_bottom = 0;

    std::vector<cv::Point> source;
    std::vector<cv::Point> target;

    for (size_t i = 0; i < len / 2; ++i)
    {
        auto top = points[i];
        auto bottom = points[len - 1 - i];
        source.push_back(cv::Point(int(top[0]), int(top[1])));
        source.push_back(cv::Point(int(bottom[0]), int(bottom[1])));
        target.push_back(cv::Point(int(top[0]), int(y_top)));
        target.push_back(cv::Point(int(top[0]), int(y_bottom)));
        if (top[0] > x_bottom)
        {
            x_bottom = top[0];
        }
    }

    std::vector<cv::DMatch> matches;
    for (int i = 0; i < len; ++i)
    {
        matches.push_back(cv::DMatch(i, i, 0));
    }
    const auto tps = cv::createThinPlateSplineShapeTransformer();
    // std::cout << "tps_img: step1" << std::endl;
    tps->estimateTransformation(target, source, matches);

    cv::Mat img;
    tps->warpImage(src_image, img);
    tps_target = target;
    // std::cout << "tps_img: end" << std::endl;
    return img;
}

template <typename T1 = double, typename T2=cv::Point>
std::vector<T2> vec2points(const std::vector<std::vector<T1>>& points)
{
    std::vector<T2> result;
    for (const auto& p : points)
    {
        // 假设每个内部向量有两个元素，分别代表x和y坐标  
        if (p.size() >= 2)
        {
            result.emplace_back(p[0], p[1]); // 将整数转换为float并添加到结果向量中  
        }
    }
    return result;
}

template <typename T1 = cv::Point, typename T2=double>
std::vector<std::vector<T2>> points2vec(const std::vector<T1>& points)
{
    std::vector<std::vector<T2>> result;
    for (const auto& p : points)
    {
        result.push_back({T2(p.x), T2(p.y)});
    }
    return result;
}

cv::Mat crop(const std::vector<std::vector<int>>& raw_points, const cv::Mat& img, const int padding)
{
    std::vector<std::vector<double>> points = convert_vec_type<int, double>(raw_points);
    auto img_width = img.cols;
    auto img_height = img.rows;
    if(is_debug) std::cout << "img shape: " << img_width << "*" << img_height << std::endl;
    
    // 最小矩形
    std::vector<std::vector<double>> raw_min_box;
    auto raw_min_box_center = get_minarea_sorted_rect(points, raw_min_box);
    auto diagonal_len = points_len(raw_min_box[0], raw_min_box[2]);
    if(is_debug) std::cout << "diagonal_len: " << diagonal_len << std::endl;
    if (diagonal_len < img_width / 15.0)
    {
        return cv::Mat{};
    }
    if (diagonal_len < img_width / 12.0)
    {
        return Utility::GetMinareaRectCrop(img, raw_points);
    }
    const auto raw_min_box_tu = tu_point(raw_min_box, points);
    
    // 最小外接圆
    cv::Mat gray = cv::Mat::zeros(img.size(), CV_8UC3);
    std::vector<std::vector<cv::Point>> draw_contours = {vec2points(points)};
    // cv::drawContours(temp_img, draw_contours, -1, cv::Scalar(255, 255, 255), 2);
    cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 3);
    draw_points(gray, points, cv::Scalar(255, 255, 255));
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, 100, 100, 3, img_width/12, std::min(img_width,1200));
    std::vector<double> distances;
    if(circles.size()==0)
    {
        return cv::Mat{};
    }
    for (const auto& circle : circles)
    {
        double distance = 0;
        cv::Vec3f c = circle;
        std::vector<double> center = {c[0], c[1]};
        const double radius = c[2];
        for (auto p : points)
        {
            auto d = point_2_circle_distance({p[0], p[1]}, center, radius);
            distance += d > 0 ? d : -d*100;
        }
        if ((raw_min_box_tu[0] - raw_min_box_center[0]) * (raw_min_box_tu[0] - center[0]) < 0 || (
           raw_min_box_tu[1] - raw_min_box_center[1]) * (raw_min_box_tu[1] - center[1]) < 0)
        {
           distance = 1000000;
        }
        if(radius<diagonal_len)
        {
            distance = 1000000;
        }
        distances.push_back(distance);
        // if(is_debug) std::cout << "circle: " << distance << ", " << radius << std::endl;
    }
    int index = 0;
    double min_distance = DBL_MAX;
    int i = 0;
    for (const auto distance : distances)
    {
        if (distance > 0 && distance < min_distance)
        {
            min_distance = distance;
            index = i;
        }
        i++;
    }
    if(is_debug) std::cout << "min_distance: " << min_distance << std::endl;
    
    // 外圆圆心
    std::vector<double> circle_center = {circles[index][0], circles[index][1]};
    circle_center = offset_circle_center(raw_min_box, circle_center);
    auto out_radius = circles[index][2];
    if(is_debug)
    {
        std::cout << "circle_center: " << circle_center[0] << "," << circle_center[1] << std::endl;
        std::cout << "out_radius: " << out_radius << std::endl;
    }
    
    // 线段(圆心和矩心)与Y轴的夹角
    auto angle_y = angle_with_y_axis(circle_center, raw_min_box_center);
    if(is_debug) std::cout << "angle: " << angle_y << std::endl;
    
    // 旋转后的min_box
    std::vector<std::vector<double>> min_box;
    for (const auto& p : raw_min_box)
    {
        auto new_point = rotate_point(p, circle_center, angle_y);
        min_box.push_back(new_point);
    }
    auto min_box_center = rotate_point(raw_min_box_center, circle_center, angle_y);
    
    // min_box 点位排序
    min_box = min_box_sort(min_box, min_box_center);
    auto min_box_width = points_len(min_box[0], min_box[3]);
    auto min_box_height = points_len(min_box[0], min_box[1]);
    if(is_debug) std::cout << "raw_min_box shape: " << min_box_width << "*" << min_box_height << std::endl;

    // 扩大外圆
    auto diff_y = circle_center[1] - min_box[0][1] - out_radius;
    out_radius += diff_y;
    if(is_debug)
    {
        cv::circle(gray, cv::Point(int(circle_center[0]), int(circle_center[1])), int(out_radius),
               cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
    }
    
    // 内圈半径
    auto in_radius = get_in_circle(points, circle_center);
    if(is_debug)
    {
        draw_points(gray, min_box, cv::Scalar(255, 0, 0), true);
        std::vector<std::vector<cv::Point>> _t = {vec2points(min_box)};
        cv::drawContours(gray, _t, -1, cv::Scalar(255, 255, 255), 1);
        draw_points(gray, raw_min_box, cv::Scalar(255, 0, 0), true);
        _t = {vec2points(raw_min_box)};
        cv::drawContours(gray, _t, -1, cv::Scalar(255, 255, 255), 1);
        cv::circle(gray, cv::Point(int(circle_center[0]), int(circle_center[1])), int(in_radius), cv::Scalar(255, 255, 255),
                   3, cv::LINE_AA);
        std::cout << "in_radius: " << in_radius << std::endl;
    }
    
    // 圆弧分段
    auto out_circle_left = point_on_circle(min_box[1], circle_center, out_radius);
    auto out_circle_right = point_on_circle(min_box[2], circle_center, out_radius);
    auto in_circle_left = point_on_circle(min_box[1], circle_center, in_radius);
    auto in_circle_right = point_on_circle(min_box[2], circle_center, in_radius);
    auto out_segments = split_arc_into_segments(out_circle_left, out_circle_right, circle_center, out_radius, 11);
    auto in_segments = split_arc_into_segments(in_circle_left, in_circle_right, circle_center, in_radius, 11);
    
    // min_box 扩大
    auto out_circle_len = calculate_arc_length(out_segments[0], out_segments.back(), circle_center, out_radius);
    auto padding_x = (out_circle_len - min_box_width) / 2;
    auto padding_y = std::max(in_segments[0][1], in_segments.back()[1]) - min_box[1][1] + padding;
    min_box = expend_min_box(min_box, padding_x, padding_y);
    min_box_width += padding_x * 2;
    min_box_height += padding_y * 2;
    if(is_debug)
    {
        std::vector<std::vector<cv::Point>> _t = {vec2points(min_box)};
        cv::drawContours(gray, _t, -1, cv::Scalar(255, 255, 255), 1);
        draw_points(gray, min_box, cv::Scalar(255, 0, 0), true);
    }
    
    // raw_min_box 扩大
    i = 0;
    for (int i = 0; i < 4; ++i)
    {
        raw_min_box[i] = rotate_point(min_box[i], circle_center, 360 - angle_y);
    }
    if(is_debug)
    {
        std::vector<std::vector<cv::Point>> _t = {vec2points(raw_min_box)};
        cv::drawContours(gray, _t, -1, cv::Scalar(255, 255, 255), 1);
        draw_points(gray, raw_min_box, cv::Scalar(255, 0, 0), true);
    }
    
    // tps映射点
    auto tps_points = out_segments;
    tps_points.resize(tps_points.size() + in_segments.size());
    std::copy(in_segments.rbegin(), in_segments.rend(), tps_points.begin() + out_segments.size());
    if(is_debug) draw_points(gray, tps_points, cv::Scalar(255, 0, 0));
    
    // 平移
    auto diff_x = min_box[0][0];
    diff_y = min_box[0][1];
    for (auto& tps_point : tps_points)
    {
        tps_point[0] -= diff_x;
        tps_point[1] -= diff_y;
    }
    
    // 截取最小图
    auto min_img = get_min_img(img, vec2points<double, cv::Point2f>(raw_min_box), min_box_width, min_box_height);
    
    // tps
    std::vector<cv::Point> tps_targets;
    min_img = tps_img(tps_points, min_img, tps_targets);
    
    // 高度剪裁
    auto min_target_y = 10000;
    auto max_target_y = 0;
    for (const auto& p : tps_targets)
    {
        if (p.y < min_target_y)
            min_target_y = p.y;
        if (p.y > max_target_y)
            max_target_y = p.y;
    }
    auto min_img_size = min_img.size();
    auto crop_y_top = std::max(min_target_y - double(padding), 0.0);
    auto crop_y_bottom = std::min(max_target_y + double(padding), double(min_img_size.height));
    cv::Rect rect(0, int(crop_y_top), min_img_size.width, int(crop_y_bottom) - int(crop_y_top));
    auto crop_img = min_img(rect);

    // 缩小
    if (min_img_size.width > 250)
    {
        auto rate = 250.0 / min_img_size.width;
        cv::resize(crop_img, crop_img, cv::Size(250, int(crop_img.size().height * rate)));
    }
    

    if(is_debug) cv::imwrite("F:\\test.jpg", gray);
    return crop_img;
}

std::vector<cv::Mat> predict_split(const cv::Mat& img)
{
    std::vector<cv::Mat> img_list;

    const auto det_boxes = det(img);

    for (const auto& boxes : det_boxes)
    {
        cv::Mat crop_img;
        //crop_img = Utility::GetRotateCropImage(img, ocr_result[j].box);
        crop_img = Utility::GetMinareaRectCrop(img, boxes);
        img_list.push_back(crop_img);
    }

    return img_list;
}

std::vector<OCRPredictResult> predict(const cv::Mat& img,const int padding)
{
    const auto det_boxes = det(img);
    std::vector<cv::Mat> img_list;

    // args use_dilation 合并轮廓
    cv::Mat temp_img = cv::Mat::zeros(img.size(), CV_8UC1);
    // cvtColor(temp_img, temp_img, cv::COLOR_BGR2GRAY);
    cv::threshold(temp_img, temp_img, 76.5, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> bbb;
    for (const auto& boxes : det_boxes)
    {
        // draw_points(temp_img,convert_vec_type<int,double>(boxes),cv::Scalar(255,255,255),false);
        bbb.push_back(vec2points(boxes));
    }
    cv::drawContours(temp_img, bbb, -1, cv::Scalar(255, 255, 255), 2);
    
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(temp_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if(is_debug)
    {
        const cv::Mat result_img = img.clone();
        cv::drawContours(result_img, contours, -1, cv::Scalar(255, 255, 255), 2);
        // for (const auto& boxes : det_boxes)
        // {
        //     draw_points(result_img,convert_vec_type<int,double>(boxes),cv::Scalar(255,255,255),false);
        // }
        cv::imwrite("F:\\test_contours.jpg",result_img);
    }
    
    
    int i = 0;
    // std::cout << "det_boxes: " << det_boxes.size() << std::endl;
    for (const auto& boxes : contours)
    {
        // std::cout << "============det_box: " << i << std::endl;
        // print_points(boxes);
        auto crop_img = crop(points2vec<cv::Point,int>(boxes), img, padding);
        // auto crop_img = crop(boxes, img, padding);
        if (!crop_img.empty())
        {
            // img_list.push_back(crop_img);
            // 宽高比<5，就有可能是多行
            const auto crop_img_shape = crop_img.size();
            if(is_debug) std::cout << "============crop_img shape: " << crop_img_shape.width << "*" << crop_img_shape.height << std::endl;
            if(crop_img_shape.width*1.0/crop_img_shape.height < 5)
            {
                const auto split_imgs = predict_split(crop_img);
                // 多个框才加入队列
                if(split_imgs.size()>1)
                {
                    for(const auto& split_img : split_imgs)
                    {
                        img_list.emplace_back(split_img);
                    }
                }
                
            }
            img_list.emplace_back(crop_img);
        }
        i++;
        // if(i>1) break;
    }
    if(is_debug)
    {
        i = 0;
        for(const auto& r_img : img_list)
        {
            cv::imwrite("F:\\crop" + std::to_string(i) + ".jpg", r_img);
            i++;
        }
    }
    return rec(img_list);
}


int main(int argc, char** argv)
{
    // Parsing command-line
    //google::ParseCommandLineFlags(&argc, &argv, true);

    //std::vector<cv::String> cv_all_img_names;
    //cv::glob(FLAGS_image_dir, cv_all_img_names);
    //// std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

    //if (!Utility::PathExists(FLAGS_output))
    //{
    //    Utility::CreateDir(FLAGS_output);
    //}

    //const cv::Mat img = cv::imread(cv_all_img_names[0], cv::IMREAD_COLOR);

    std::string path_dir;
    while (std::cin >> path_dir) {
        const cv::Mat img = cv::imread(path_dir, cv::IMREAD_COLOR);
        const std::vector<OCRPredictResult> ocr_result = predict(img, 5);
        // const std::vector<OCRPredictResult> ocr_result = rec({img});

        std::cout << "result: " << get_result_json(ocr_result) << std::endl;
    }

    
}


char* ocr_mat(const cv::Mat& img,const int padding = 5)
{
    const auto ocr_results = predict(img,padding);
    return get_result_json(ocr_results);
}

// 修改args.cpp参数
extern "C" _declspec(dllexport) char* Ocr(const char* image_dir,const int padding)
{
    const cv::Mat img = cv::imread(image_dir, cv::IMREAD_COLOR);
    return ocr_mat(img, padding);
}

extern "C" _declspec(dllexport) char* OcrMat(const cv::Mat* img,const int padding)
{
    return ocr_mat(*img,padding);
}
