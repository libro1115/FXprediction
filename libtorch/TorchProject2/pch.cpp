// pch.cpp: プリコンパイル済みヘッダーに対応するソース ファイル

//#include "pch.h"
#include "framework.h"
#include"FXModel.h"
// プリコンパイル済みヘッダーを使用している場合、コンパイルを成功させるにはこのソース ファイルが必要です。
/********************************************************************************************************************
//文字列をstringに変換
*********************************************************************************************************************/
std::string tostring(const wchar_t* c) {

	std::string tmps;

	if (c == nullptr)
		return tmps;

	size_t sz = wcslen(c);

	tmps.reserve(sz * 2);//メモリを確保し、newが走りすぎないようにする

	const size_t CNT_MAX = 100;
	char tmpc[CNT_MAX];
	wchar_t tmpw[CNT_MAX];

	const wchar_t* p = c;
	while (*p) {
		//サロゲートペアなら2文字分、違うなら１文字分だけtmpwに確保
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
		wcstombs(tmpc, tmpw, CNT_MAX);//tmpwの内容を変換してtmpcに代入
		tmps += tmpc;
	}

	return tmps;
}

/********************************************************************************************************************
//推論
*********************************************************************************************************************/
_DLLAPI void __stdcall Predict(wchar_t* path, double* data, int data_size, double* ans_address)
{
	auto model = FXModel(tostring(path));//モデル読み込み
	auto tensor = torch::zeros({ 1,data_size });
	//Xデータをtensorに変換
	for (int i = 0; i < data_size; i++)
	{
		tensor[0][i] = *(data + i);
	}
	auto ans = model.Predict(tensor);
	for (int i = 0; i < 7; i++)
		ans_address[i] = 999;//読み込み失敗

	//急降下
	ans_address[0] = ans[0][0].item<double>();
	ans_address[0] = ans[0][1].item<double>();
	//急上昇
	ans_address[0] = ans[1][0].item<double>();
	ans_address[0] = ans[1][1].item<double>();
	//推論
	ans_address[0] = ans[2][0].item<double>();
	ans_address[0] = ans[2][1].item<double>();
	ans_address[0] = ans[2][2].item<double>();
	return;
}