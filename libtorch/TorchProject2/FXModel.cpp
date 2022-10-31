#include "FXModel.h"
#include<vector>
FXModel::FXModel(const std::string model_path):model(torch::jit::load(model_path))
{

	model.eval();
}

FXModel::~FXModel()
{}

/********************************************************************************************************************
//Tensor���琄�_
*********************************************************************************************************************/
std::vector<at::Tensor> FXModel::Predict(at::Tensor input)
{
	auto output = this->model.forward({ input }).toTensor();
	auto ans = std::vector<at::Tensor>();
	ans.resize(3);
	ans[0] = GetSuddenDown(output);
	ans[1] = GetSuddenUpp(output);
	ans[2] = GetUpDown(output);
	return ans;
	//return std::make_tuple(GetSuddenDown(output), GetSuddenUpp(output), GetUpDown(output));
}
/********************************************************************************************************************
//�}�~�����_
*********************************************************************************************************************/
at::Tensor FXModel::GetSuddenDown(const at::Tensor &output) const
{
	return output.narrow(1, 0, 2);// .softmax(1);
}

/********************************************************************************************************************
//�}�㏸���_
*********************************************************************************************************************/
at::Tensor FXModel::GetSuddenUpp(at::Tensor output) const
{
	return output.narrow(1, 2, 2);// .softmax(1);
}

/********************************************************************************************************************
//���蔃�����_
*********************************************************************************************************************/
at::Tensor FXModel::GetUpDown(at::Tensor output) const
{
	return output.narrow(1, 4, 3);// .softmax(1);
}

