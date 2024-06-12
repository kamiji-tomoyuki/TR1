//openCvをインクルード
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// 特徴抽出 & 画像生成
Mat CreateImages(const Mat& image);

int main() {
	// 画像を読み込む
	Mat A = imread("image/kirby.jpg");     //カービィ　
	Mat B = imread("image/uvChecker.png"); //UV
	Mat C = imread("image/map.png");       //マップチップ

	// 読み取らせる画像
	Mat image = C;

	// 特徴を学習 & 新しい画像 を生成
	Mat newImage = CreateImages(image);

	// 元の画像 & 作った画像 を描画！！
	imshow("Image", image);
	imshow("newImage", newImage);

	waitKey(0);//画像を描画させるために必要
	return 0;
}

Mat CreateImages(const Mat& image) {
	// グレースケールに変換 -> SIFTの処理を簡単に
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	// 特徴量を求める
	Ptr<SIFT> sift = SIFT::create();
	vector<KeyPoint> keyPoint;
	Mat descriptors;//特徴点
	sift->detectAndCompute(grayImage, noArray(), keyPoint, descriptors);

	// 新しい画像を生成！！ (TR1_1回目は特徴点を描画する)
	Mat newImage = image.clone();
	drawKeypoints(image, keyPoint, newImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	return newImage;
}