// pch.cpp: �v���R���p�C���ς݃w�b�_�[�ɑΉ�����\�[�X �t�@�C��

//#include "pch.h"
#include "framework.h"
#include"FXModel.h"
// �v���R���p�C���ς݃w�b�_�[���g�p���Ă���ꍇ�A�R���p�C���𐬌�������ɂ͂��̃\�[�X �t�@�C�����K�v�ł��B
/********************************************************************************************************************
//�������string�ɕϊ�
*********************************************************************************************************************/
std::string tostring(const wchar_t* c) {

	std::string tmps;

	if (c == nullptr)
		return tmps;

	size_t sz = wcslen(c);

	tmps.reserve(sz * 2);//���������m�ۂ��Anew�����肷���Ȃ��悤�ɂ���

	const size_t CNT_MAX = 100;
	char tmpc[CNT_MAX];
	wchar_t tmpw[CNT_MAX];

	const wchar_t* p = c;
	while (*p) {
		//�T���Q�[�g�y�A�Ȃ�2�������A�Ⴄ�Ȃ�P����������tmpw�Ɋm��
		if (IS_HIGH_SURROGATE(*p) == true) {
			wcsncpy(tmpw, p, 2);
			tmpw[2] = L'\0';
			p += 2;
		}
		else {
			wcsncpy(tmpw, p, 1);
			tmpw[1] = L'\0';
			p += 1;
		}
		wcstombs(tmpc, tmpw, CNT_MAX);//tmpw�̓��e��ϊ�����tmpc�ɑ��
		tmps += tmpc;
	}

	return tmps;
}

/********************************************************************************************************************
//���_
*********************************************************************************************************************/
_DLLAPI void __stdcall Predict(wchar_t* path, double* data, int data_size, double* ans_address)
{
	auto model = FXModel(tostring(path));//���f���ǂݍ���
	auto tensor = torch::zeros({ 1,data_size });
	//X�f�[�^��tensor�ɕϊ�
	for (int i = 0; i < data_size; i++)
	{
		tensor[0][i] = *(data + i);
	}
	auto ans = model.Predict(tensor);
	for (int i = 0; i < 7; i++)
		ans_address[i] = 999;//�ǂݍ��ݎ��s

	//�}�~��
	ans_address[0] = ans[0][0].item<double>();
	ans_address[0] = ans[0][1].item<double>();
	//�}�㏸
	ans_address[0] = ans[1][0].item<double>();
	ans_address[0] = ans[1][1].item<double>();
	//���_
	ans_address[0] = ans[2][0].item<double>();
	ans_address[0] = ans[2][1].item<double>();
	ans_address[0] = ans[2][2].item<double>();
	return;
}