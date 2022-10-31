#pragma once


#include<torch/torch.h>
#include<torch/script.h>
#include<iostream>
#include<vector>
#include<atlstr.h>
class FXModel
{
public:
	FXModel(const std::string model_path);
	~FXModel();
	std::vector<at::Tensor> Predict(at::Tensor input);

protected:
	torch::jit::Module model;
	at::Tensor GetSuddenDown(const at::Tensor &output)const;
	at::Tensor GetSuddenUpp(at::Tensor output)const;
	at::Tensor GetUpDown(at::Tensor output)const;
private:
};

//extern "C" DLL FXModel * CreateInstance(void);