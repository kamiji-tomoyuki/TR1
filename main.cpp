#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

// 複数の画像から特徴を元に新しい画像を生成する関数
Mat CreateImages(const vector<Mat>& images);

int main() {
    // 複数の画像をベクターに読み込む
    vector<Mat> images;
    images.push_back(imread("image/map1.png"));//赤      
    images.push_back(imread("image/map2.png"));//青      
    images.push_back(imread("image/map3.png"));//緑     
    images.push_back(imread("image/map4.png"));//シアン     

    // 複数の画像から特徴を元に新しい画像を生成
    Mat newImage = CreateImages(images);

    // 元の画像と生成した画像を表示
    for (size_t i = 0; i < images.size(); ++i) {
        string imageName = "Image " + to_string(i + 1);
        imshow(imageName, images[i]);
    }
    imshow("New Image", newImage);

    waitKey(0); // ウィンドウ内でキー入力を待機
    return 0;
}

Mat CreateImages(const vector<Mat>& images) {
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> allKeypoints;
    Mat descriptors;

    // 各画像の色を定義
    vector<Scalar> colors = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 255, 0) };

    // 新しい画像用の空のキャンバスを作成　
    Mat newImage = Mat::zeros(images[0].size(), images[0].type());

    // 各画像について特徴を抽出
    for (size_t i = 0; i < images.size(); ++i) {
        Mat grayImage;
        cvtColor(images[i], grayImage, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat imageDescriptors;
        sift->detectAndCompute(grayImage, noArray(), keypoints, imageDescriptors);

        // この画像からのキーポイントを描画
        for (const KeyPoint& kp : keypoints) {
            Point center(cvRound(kp.pt.x), cvRound(kp.pt.y));
            int radius = cvRound(kp.size / 2);
            // 画像に対応する色を使用して特徴点を描画
            circle(newImage, center, radius, colors[i % colors.size()], 1, LINE_AA);
        }

        // この画像からのキーポイントと記述子を追加
        allKeypoints.insert(allKeypoints.end(), keypoints.begin(), keypoints.end());

        if (descriptors.empty()) {
            descriptors = imageDescriptors.clone();
        }
        else {
            vconcat(descriptors, imageDescriptors, descriptors);
        }
    }

    return newImage;
}