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
    images.push_back(imread("image/kirby.jpg"));     // カービィ
    images.push_back(imread("image/uvChecker.png")); // UV
    images.push_back(imread("image/map.png"));       // マップチップ

    // すべての画像が正しく読み込まれていることを確認
    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].empty()) {
            cout << "画像 " << i << " を開くことができませんでした。" << endl;
            return -1;
        }
    }

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

    // 各画像について特徴を抽出
    for (const Mat& image : images) {
        Mat grayImage;
        cvtColor(image, grayImage, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat imageDescriptors;
        sift->detectAndCompute(grayImage, noArray(), keypoints, imageDescriptors);

        // この画像からのキーポイントと記述子を追加
        allKeypoints.insert(allKeypoints.end(), keypoints.begin(), keypoints.end());

        if (descriptors.empty()) {
            descriptors = imageDescriptors.clone();
        }
        else {
            vconcat(descriptors, imageDescriptors, descriptors);
        }
    }

    // 新しい画像用の空のキャンバスを作成
    Mat newImage = Mat::zeros(images[0].size(), images[0].type());

    // ランダムに新しい画像にキーポイントを描画
    for (const KeyPoint& kp : allKeypoints) {
        Point center(cvRound(kp.pt.x), cvRound(kp.pt.y));
        int radius = cvRound(kp.size / 2);
        circle(newImage, center, radius, Scalar::all(255), 1, LINE_AA);
    }

    return newImage;
}