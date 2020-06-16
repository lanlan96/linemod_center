//
// Created by 兰育青 on 2020-02-23.
//

#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include <string>
#include<ctime>


using namespace std;
using namespace cv;


//先设置这个prefix
static std::string prefix = "/Users/lyq/lyq-projects/shape_based_matching/test/";

clock_t start,finish;


cv::Mat Resize(cv::Mat src,float scale) {

    cv::Mat dst;

    float scaleW = scale;
    //定义图像的大小，宽度缩小80%
    float scaleH = scaleW;
    //定义图像的大小，高度缩小80%

    int width = static_cast<float>(src.cols*scaleW);
    //定义想要扩大或者缩小后的宽度，src.cols为原图像的宽度，乘以80%则得到想要的大小，并强制转换成float型
    int height = static_cast<float>(src.rows*scaleH);
    //定义想要扩大或者缩小后的高度，src.cols为原图像的高度，乘以80%则得到想要的大小，并强制转换成float型

    resize(src, dst, cv::Size(width, height));
    //重新定义大小的函数


    return dst;
}

// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
    namespace
    {

        template <typename T>
        static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                                const std::pair<float, T>& pair2)
        {
            return pair1.first > pair2.first;
        }

    } // namespace

    inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                                 std::vector<std::pair<float, int> >& score_index_vec)
    {
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] > threshold)
            {
                score_index_vec.push_back(std::make_pair(scores[i], i));
            }
        }
        std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                         SortScorePairDescend<int>);
        if (top_k > 0 && top_k < (int)score_index_vec.size())
        {
            score_index_vec.resize(top_k);
        }
    }

    template <typename BoxType>
    inline void NMSFast_(const std::vector<BoxType>& bboxes,
                         const std::vector<float>& scores, const float score_threshold,
                         const float nms_threshold, const float eta, const int top_k,
                         std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
    {
        CV_Assert(bboxes.size() == scores.size());
        std::vector<std::pair<float, int> > score_index_vec;
        GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

        // Do nms.
        float adaptive_threshold = nms_threshold;
        indices.clear();
        for (size_t i = 0; i < score_index_vec.size(); ++i) {
            const int idx = score_index_vec[i].second;
            bool keep = true;
            for (int k = 0; k < (int)indices.size() && keep; ++k) {
                const int kept_idx = indices[k];
                float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            }
            if (keep)
                indices.push_back(idx);
            if (keep && eta < 1 && adaptive_threshold > 0.5) {
                adaptive_threshold *= eta;
            }
        }
    }


// copied from opencv 3.4, not exist in 3.0
    template<typename _Tp> static inline
    double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
        _Tp Aa = a.area();
        _Tp Ab = b.area();

        if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
            // jaccard_index = 1 -> distance = 0
            return 0.0;
        }

        double Aab = (a & b).area();
        // distance = 1 - jaccard_index
        return 1.0 - Aab / (Aa + Ab - Aab);
    }

    template <typename T>
    static inline float rectOverlap(const T& a, const T& b)
    {
        return 1.f - static_cast<float>(jaccardDistance__(a, b));
    }

    void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                  const float score_threshold, const float nms_threshold,
                  std::vector<int>& indices, const float eta=1, const int top_k=0)
    {
        NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
    }

}

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
                (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};




void angle_test(string mode = "test", string case_num="case_f",string objid="56C86", string path="/realdata/train_p2.png", float range_0=1,float range_1=3,
                int num_feature =100,bool write_flag= false){


    int t1=4;
    int t2=8;

    string outrange="-"+std::to_string(int(range_0))+"-"+std::to_string(int(range_1))+"-"+std::to_string(num_feature)
                    +"-"+std::to_string(t1)+"-"+std::to_string(t2);
    string templ_path=prefix+case_num+"/"+objid+"/test_templ"+outrange+".yaml";
    string info_path=prefix + case_num+"/"+objid+"/test_info"+outrange+".yaml";
    string result_dir=prefix+case_num+"/"+objid;
    fstream _file;
    _file.open(result_dir, ios::in);
    if(!_file)
    {
        string cmd = "mkdir "+result_dir;
        system(cmd.data());
    }


    line2Dup::Detector detector(num_feature, {t1, t2});
//    mode = "test";
    if(mode == "train"){
        Mat img = imread(prefix+path);
        cvtColor(img, img, CV_BGR2GRAY);

        int pixelR, pixelG, pixelB;//像素rgb的值
        assert(!img.empty() && "check your img path");

        Rect roi(0, 0, img.cols, img.rows);
        cout<<img.cols<<" "<<img.rows<<endl;
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        /// padding to avoid rotating out
        int padding ;

        if (img.cols>img.rows){
            padding =(range_1/2)*img.cols;
        }
        else
            padding=(range_1/2)*img.rows;

        cout<<"channel: "<<img.channels()<<endl;
//        padding=0;
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
//        shapes.angle_range = {-1, 1};
//        shapes.angle_step = 1;

        shapes.scale_range = {range_0, range_1};  //模版放缩范围
        shapes.scale_step = 0.002f;

        shapes.produce_infos();  ///scale
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";
        for(auto& info: shapes.infos){
            imshow("train", shapes.src_of(info));
            waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            std::cout << "\ninfo.scale: " << info.scale << std::endl;

            int num_f=int(num_feature*info.scale);
            ///control number of features
            if (num_f<=20)
                num_f=int(num_f*2/3);
            ///
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info)
                    ,num_f);
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }


        detector.writeClasses(templ_path);
        shapes.save_infos(infos_have_templ, info_path);
        std::cout << "train end" << std::endl << std::endl;
    }

    else if(mode=="test"){

        float sum_x=0;
        float sum_y=0;
        std::vector<std::string> ids;
        ids.push_back("test");

        cout<<"reading tmplates: "<<templ_path<<endl;

        detector.readClasses(ids, templ_path);


        // angle & scale are saved here, fetched by match id
        auto infos = shape_based_matching::shapeInfo_producer::load_infos(info_path);

        cout<<info_path<<endl;

        Mat test_img_ori = imread(prefix+path);
        cvtColor(test_img_ori, test_img_ori, CV_BGR2GRAY);

        Mat test_img=test_img_ori;

        cout<<"test_img: "<<test_img.cols<<" "<<test_img.rows<<endl;

        int pixelR, pixelG, pixelB;//像素rgb的值
        assert(!test_img.empty() && "check your img path");

        int padding = 0;
        cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,
                                     test_img.cols + 2*padding, test_img.type(), cv::Scalar(0));
        test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

        int stride = 16;
        int n = padded_img.rows/stride;
        int m = padded_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = padded_img(roi).clone();
        assert(img.isContinuous());

        Timer timer;
        auto matches = detector.match(img, 50, ids);  //匹配度界限
        timer.out();

        if(img.channels() == 1) cvtColor(img, img, CV_GRAY2BGR);

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 5;
        if(top5>matches.size()) top5=matches.size();

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> idxs;
        for(auto match: matches){
            Rect box;
            box.x = match.x;
            box.y = match.y;

            auto templ = detector.getTemplates("test",
                                               match.template_id);

            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
//        cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);
        cv_dnn::NMSBoxes(boxes, scores, 0, 0.1f, idxs); //nms抑制

//        for(int idx=0;idx<top5;idx++){
        for(auto idx: idxs) {

            auto match = matches[idx];
                auto templ = detector.getTemplates("test",
                                                   match.template_id);

                float r_scaled = 270 / 2.0f * infos[match.template_id].scale;

                cout << "angle: " << infos[match.template_id].angle << " " << " scale: "
                     << infos[match.template_id].scale << endl;

                // scaling won't affect this, because it has been determined by warpAffine
                // cv::warpAffine(src, dst, rot_mat, src.size()); last param
                float train_img_half_width = 270 / 2.0f + 100;

                // center x,y of train_img in test img
                float x = match.x - templ[0].tl_x + train_img_half_width;
                float y = match.y - templ[0].tl_y + train_img_half_width;

                cv::Vec3b randColor;
                randColor[0] = 0;
                randColor[1] = 0;
                randColor[2] = 255;
                cout << "feature size :  " << templ[0].features.size() << endl;

                //feat.x feat.y 为模版左上角在测试图像中的像素坐标

                for (int i = 0; i < templ[0].features.size(); i++) {
                    auto feat = templ[0].features[i];
                    cv::circle(img, {feat.x + match.x, feat.y + match.y}, 3, randColor, -1);
                }
                cv::putText(img, to_string(int(round(match.similarity))),
                            Point(match.x + r_scaled - 10, match.y - 3), FONT_HERSHEY_PLAIN, 2, randColor);

                std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
                std::cout << "match.similarity: " << match.similarity << std::endl;
            }
//        }
//            imshow("img", img);
            if (write_flag == true) {
                cv::imwrite("../test/case_test/result/result.jpg", img);
            }

            std::cout << "test end" << std::endl << std::endl;

    }
}



void MIPP_test(){
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType                  << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType              << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion               << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit       << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes                            << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit    ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
    std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
    std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl << std::endl;
}


int main(){
    srand((unsigned) time(NULL));//diff color
    MIPP_test();

    start=clock();
    cout << "开始计算时间 .... " << endl;


    angle_test("train","case_test","1th","case_test/train_img/WechatIMG27_2.png",
               0.95,1.05 ,40); // train

    angle_test("test","case_test","1th","case_test/test_img/WechatIMG27.png",
               0.95,1.05,40, true); // test


    finish=clock();
    cout << "time: "<<(double)(finish-start)/ CLOCKS_PER_SEC   << " (s) "<< endl;
}
