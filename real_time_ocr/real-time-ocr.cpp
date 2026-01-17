//
// Created by 袁志鹏 on 2026/1/17.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include "camera.h"
#include "ocr.h"
#include "rga.h" // EasyEAI 提供的 RGA 显示库

using namespace cv;
using namespace std;

#define CAMERA_WIDTH 720
#define CAMERA_HEIGHT 1280
#define ROTATE 270
#define IMGRATIO 3
#define IMAGE_SIZE (CAMERA_WIDTH * CAMERA_HEIGHT * IMGRATIO)

#define THRESHOLD 0.3
#define BOX_THRESHOLD 0.9
#define USE_DILATION false
#define DB_UNCLIP_RATIO 1.5

int main() {
    int ret;

    // 1️⃣ 初始化摄像头
    ret = rgbcamera_init(CAMERA_WIDTH, CAMERA_HEIGHT, ROTATE);
    if (ret) {
        printf("Camera init failed!\n");
        return -1;
    }

    char* pbuf = (char*)malloc(IMAGE_SIZE);
    if (!pbuf) {
        printf("malloc failed\n");
        rgbcamera_exit();
        return -1;
    }

    // 2️⃣ 初始化 RGA
    rga_init();

    // 3️⃣ 初始化 OCR 模型
    rknn_app_context_t ocr_det_ctx, ocr_rec_ctx;
    memset(&ocr_det_ctx, 0, sizeof(rknn_app_context_t));
    memset(&ocr_rec_ctx, 0, sizeof(rknn_app_context_t));

    ret = ocr_det_init("ocr_det.model", &ocr_det_ctx);
    if (ret != 0) {
        printf("OCR detection model init failed\n");
    }
    ret = ocr_rec_init("ocr_ret.model", &ocr_rec_ctx);
    if (ret != 0) {
        printf("OCR recognition model init failed\n");
    }

    ocr_det_postprocess_params params;
    params.threshold = THRESHOLD;
    params.box_threshold = BOX_THRESHOLD;
    params.use_dilate = USE_DILATION;
    params.db_score_mode = (char*)"slow";
    params.db_box_type = (char*)"poly";
    params.db_unclip_ratio = DB_UNCLIP_RATIO;

    printf("Starting real-time camera + OCR display...\n");

    while (true) {
        // 4️⃣ 获取摄像头一帧
        ret = rgbcamera_getframe(pbuf);
        if (ret) {
            printf("Failed to get frame\n");
            break;
        }

        // 转成 OpenCV Mat
        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, pbuf);

        // 5️⃣ OCR 检测
        ocr_det_result results;
        ret = ocr_det_run(&ocr_det_ctx, frame, &params, &results);
        if (ret == 0) {
            // 6️⃣ 遍历检测结果，识别文字并画框
            for (int i = 0; i < results.count; i++) {
                // 画框
                line(frame, Point(results.box[i].left_top.x, results.box[i].left_top.y),
                     Point(results.box[i].right_top.x, results.box[i].right_top.y),
                     Scalar(0,255,0), 2);
                line(frame, Point(results.box[i].right_top.x, results.box[i].right_top.y),
                     Point(results.box[i].right_bottom.x, results.box[i].right_bottom.y),
                     Scalar(0,255,0), 2);
                line(frame, Point(results.box[i].right_bottom.x, results.box[i].right_bottom.y),
                     Point(results.box[i].left_bottom.x, results.box[i].left_bottom.y),
                     Scalar(0,255,0), 2);
                line(frame, Point(results.box[i].left_bottom.x, results.box[i].left_bottom.y),
                     Point(results.box[i].left_top.x, results.box[i].left_top.y),
                     Scalar(0,255,0), 2);

                // 截取文字区域
                Mat crop_img = GetRotateCropImage(frame, results.box[i]);

                // OCR 识别
                ocr_rec_result rec_results;
                ocr_rec_run(&ocr_rec_ctx, crop_img, &rec_results);

                // 在画面上写文字
                putText(frame, rec_results.str,
                        Point(results.box[i].left_top.x, results.box[i].left_top.y - 5),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);
            }
        }

        // 7️⃣ 用 RGA 显示到屏幕
        rga_blit_to_fb(frame.data, CAMERA_WIDTH, CAMERA_HEIGHT);

        // 8️⃣ 检测退出
        char key = getchar();
        if (key == 'q' || key == 'Q') break;
    }

    // 退出清理
    rga_exit();
    free(pbuf);
    rgbcamera_exit();
    ocr_det_exit(&ocr_det_ctx);
    ocr_rec_exit(&ocr_rec_ctx);

    return 0;
}
