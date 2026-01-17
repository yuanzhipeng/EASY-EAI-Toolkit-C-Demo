#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include "ocr.h"

using namespace cv;
using namespace std;

#define THRESHOLD 0.3
#define BOX_THRESHOLD 0.9
#define USE_DILATION false
#define DB_UNCLIP_RATIO 1.5

int main()
{
    // 打开摄像头
    VideoCapture cap(0); // 0 是默认摄像头
    if (!cap.isOpened()) {
        cout << "Error: Could not open camera" << endl;
        return -1;
    }

    // 初始化 OCR 模型
    rknn_app_context_t ocr_det_ctx, ocr_rec_ctx;
    memset(&ocr_det_ctx, 0, sizeof(rknn_app_context_t));
    memset(&ocr_rec_ctx, 0, sizeof(rknn_app_context_t));

    if (ocr_det_init("ocr_det.model", &ocr_det_ctx) != 0 ||
        ocr_rec_init("ocr_ret.model", &ocr_rec_ctx) != 0) {
        cout << "Error: OCR model load failed" << endl;
        return -1;
    }

    Mat frame, rgb_frame;
    while (true) {
        cap >> frame; // 获取一帧
        if (frame.empty()) break;

        cvtColor(frame, rgb_frame, COLOR_BGR2RGB);

        // OCR 检测
        ocr_det_result results;
        ocr_det_postprocess_params params;
        params.threshold = THRESHOLD;
        params.box_threshold = BOX_THRESHOLD;
        params.use_dilate = USE_DILATION;
        params.db_score_mode = (char*)"slow";
        params.db_box_type = (char*)"poly";
        params.db_unclip_ratio = DB_UNCLIP_RATIO;

        int ret = ocr_det_run(&ocr_det_ctx, rgb_frame, &params, &results);
        if (ret != 0) {
            cout << "OCR detection failed" << endl;
            continue;
        }

        // 遍历检测框
        for (int i = 0; i < results.count; i++) {
            // 画检测框
            line(frame, Point(results.box[i].left_top.x, results.box[i].left_top.y),
                 Point(results.box[i].right_top.x, results.box[i].right_top.y), Scalar(0,255,0), 1);
            line(frame, Point(results.box[i].right_top.x, results.box[i].right_top.y),
                 Point(results.box[i].right_bottom.x, results.box[i].right_bottom.y), Scalar(0,255,0), 1);
            line(frame, Point(results.box[i].right_bottom.x, results.box[i].right_bottom.y),
                 Point(results.box[i].left_bottom.x, results.box[i].left_bottom.y), Scalar(0,255,0), 1);
            line(frame, Point(results.box[i].left_bottom.x, results.box[i].left_bottom.y),
                 Point(results.box[i].left_top.x, results.box[i].left_top.y), Scalar(0,255,0), 1);

            // OCR 识别
            Mat crop_img = GetRotateCropImage(rgb_frame, results.box[i]);
            ocr_rec_result rec_res;
            ocr_rec_run(&ocr_rec_ctx, crop_img, &rec_res);

            // 显示文字在框上
            putText(frame, rec_res.str, Point(results.box[i].left_top.x, results.box[i].left_top.y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 1);
        }

        // 显示结果
        imshow("OCR Camera", frame);

        // 按 q 键退出
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
