#include <eigen3/Eigen/Dense>
#include <cmath>
#include <random>

void hat_so3(const Eigen::VectorXf &so3, Eigen::MatrixXf &so3_hat)
{
    so3_hat = Eigen::MatrixXf::Zero(3, 3);
    so3_hat(0, 1) = -so3(2);
    so3_hat(0, 2) = so3(1);
    so3_hat(1, 2) = -so3(0);
    so3_hat(1, 0) = so3(2);
    so3_hat(2, 1) = -so3(1);
    so3_hat(2, 2) = so3(0);
}

void vee_so3(const Eigen::MatrixXf &so3_hat, Eigen::VectorXf &so3)
{
    // normalize matrix.
    // this is property of skew symmetric matrix.
    Eigen::MatrixXf n = (so3_hat - so3_hat.transpose()) / 2;
    so3(0) = -so3_hat(1, 2);
    so3(1) = so3_hat(0, 2);
    so3(2) = -so3_hat(0, 1);
}

void exp_so3(const Eigen::VectorXf &so3, Eigen::MatrixXf &SO3)
{
    // calculate theta
    const float &th = so3.norm();

    // check divide by zero
    if (FP_ZERO == std::fpclassify(th))
    {
        SO3 = Eigen::MatrixXf::Identity(3, 3);
        return;
    }

    // make so3_hat
    Eigen::MatrixXf so3_hat;
    hat_so3(so3, so3_hat);

    // do exp operation
    SO3 = Eigen::MatrixXf::Identity(3, 3) +
          sin(th) / th * so3_hat +
          (1.0 - cos(th)) / (th * th) * so3_hat * so3_hat;
}

void log_so3(const Eigen::MatrixXf &SO3, Eigen::VectorXf &so3)
{
    const float &trace = SO3.trace();
    const float &cos_th = (trace - 1.0) / 2.0;

    // cos(th) == 1.0 --> th == 0 --> sin(th) == 0.0
    if (FP_ZERO == std::fpclassify(cos_th - 1.0))
    {
        so3 = Eigen::VectorXf::Zero(3);
        return;
    }
    // cos(th) == - 1.0 --> th == PI --> sin(th) == 0.0
    // modern robotics. page 87. eq 3.58
    else if (FP_ZERO == std::fpclassify(cos_th + 1.0))
    {
        const float &a = 1.0 / std::sqrt(2.0 * (1 + SO3(2, 2)));
        so3(0) = a * SO3(0, 2);
        so3(1) = a * SO3(1, 2);
        so3(2) = a * (SO3(2, 2) + 1.0);
        return;
    }

    const float &th = std::acos(cos_th);
    vee_so3((SO3 - SO3.transpose()) * th / (2.0 * std::sin(th)), so3);
}

void hat_se3(const Eigen::VectorXf &se3, Eigen::MatrixXf &se3_hat)
{
    // Eigen::VectorXf t= se3.block(0,0,3,1);
    // Eigen::VectorXf w= se3.block(3,0,3,1);

    // make so3_hat
    Eigen::MatrixXf so3_hat;
    hat_so3(se3.block(3, 0, 3, 1), so3_hat);

    // set values
    se3_hat = Eigen::MatrixXf::Zero(4, 4);
    se3_hat.block(0, 0, 3, 3) = so3_hat;
    se3_hat.block(0, 3, 3, 1) = se3.block(0, 0, 3, 1);
}

void vee_se3(const Eigen::MatrixXf &se3_hat, Eigen::VectorXf &se3)
{
    // set translation
    se3.block(0, 0, 3, 1) = se3_hat.block(0, 3, 3, 1);

    // make so3
    Eigen::VectorXf so3;
    vee_so3(se3_hat.block(0, 0, 3, 3), so3);

    // set orientation
    se3.block(0, 3, 3, 1) = so3;
}

void exp_se3(const Eigen::VectorXf &se3, Eigen::MatrixXf &SE3)
{
    // make so3_hat
    Eigen::MatrixXf so3_hat;
    hat_so3(se3.block(3, 0, 3, 1), so3_hat);

    float th = se3.norm();

    // th == 0 , which means no motion. --> Identity.
    if (FP_ZERO == std::fpclassify(th))
    {
        SE3 = Eigen::MatrixXf::Identity(4, 4);
        return;
    }

    // make V
    const float &th_sq = th * th;
    const float &th_cb = th_sq * th;
    Eigen::MatrixXf V = Eigen::MatrixXf::Identity(3, 3) +
                        so3_hat * (1.0 - std::cos(th)) / (th_sq) +
                        so3_hat * so3_hat * (th - std::sin(th)) / (th_cb);

    // make SO3
    Eigen::MatrixXf SO3;
    exp_so3(se3.block(3, 0, 3, 1), SO3);
    SE3.block(0, 0, 3, 3) = SO3;
    SE3.block(0, 3, 3, 1) = V * se3.block(0, 0, 3, 1);
}

void log_se3(const Eigen::MatrixXf &SE3, Eigen::VectorXf &se3)
{
    // make so3
    Eigen::VectorXf so3;
    log_so3(SE3.block(0, 0, 3, 3), so3);
    const float &th = so3.norm();

    // th == 0 , which means no rotation
    // modern robotics. page 106, 3.3.3.2, algorithm case (a)
    if (FP_ZERO == std::fpclassify(th))
    {
        se3.block(0, 0, 3, 1) = SE3.block(0, 3, 3, 1).normalized();
        se3.block(3, 0, 3, 1) = Eigen::VectorXf::Zero(3);
        return;
    }

    // make so3_hat
    Eigen::MatrixXf so3_hat;
    hat_so3(so3, so3_hat);

    //make V_inv
    Eigen::MatrixXf V_inv = Eigen::MatrixXf::Identity(3, 3) -
                            so3_hat * 0.5 +
                            so3_hat * so3_hat * (1.0 - (th * std::cos(0.5 * th) / (2.0 * std::sin(0.5 * th)))) / (th * th);

    //set se3
    se3.block(0, 0, 3, 1) = V_inv * SE3.block(0, 3, 3, 1);
    se3.block(3, 0, 3, 1) = so3;
}

void adj_se3(const Eigen::MatrixXf &SE3, Eigen::MatrixXf &adj)
{
    adj.resize(6, 6);
    Eigen::MatrixXf p_hat;
    hat_so3(SE3.block(0, 3, 3, 1), p_hat);
    adj.block(0, 0, 3, 3) = SE3.block(0, 0, 3, 3);
    adj.block(3, 0, 3, 3) = p_hat * SE3.block(0, 0, 3, 3);
    adj.block(0, 3, 3, 3) = Eigen::MatrixXf::Zero(3, 3);
    adj.block(3, 3, 3, 3) = SE3.block(0, 0, 3, 3);
}

Eigen::VectorXf rand_so3()
{
    Eigen::VectorXf dir = Eigen::VectorXf::Random(3).normalized();
    const float &th = Eigen::VectorXf::Random(1)(0) * M_PI;
    return dir * th;
}
