#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <algorithm>
#include <cmath>  

using namespace std;

string chan10_2(int num,int ord,int le);//ʮ����ת������ 
string chan2_16(int * a2, int len_t);//������תʮ������ 
string chan10_16(string a10);//ʮ����תʮ������ 
void zerosort(int * tar ,string x, int be,int le); //0+���� 

int main(){
    ifstream fin("data.in");
    ofstream fout("data.txt");
    ofstream foutphy("dataphy.txt");
    ofstream foutnr("datanr.txt");
    ofstream foutr("datar.txt");
    
    string canm; //�����ĵ� 
    
	string id_10; //10����ID 
	int len_t; //data���� 

	string temp;
	
    while(getline(fin,canm)){ //���з� 
    	string mes = canm.substr(0,3);

    	if(mes == "BO_"){
    		string temp = canm.substr(4,1);
    		int len_ID; //ID���� 
    		int i;
    		for(i = 4;'0'<= temp[0] && temp[0] <= '9' ;){
    			i++;
    			temp = canm.substr(i,1);
			}
			len_ID = i - 4;
			id_10 = canm.substr(4,len_ID); //�õ�10����ID
			
			int col = 0; //ð�ŵ�λ�� 
			for(i = 0; ; i++){
				if(canm[i] == ':'){
					col = i;
					break;
				}
			}
			len_t = canm[col+2] - '0';//�õ�data���� 
			
			fout << canm << endl;
			foutphy << canm << endl;
		}
		
		if(canm.substr(0,1) != ""){
			mes = canm.substr(1,3);
		}
		if(mes == "SG_"){
			int col = 0; //ð�ŵ�λ�� 
			int i;
			for(i = 0; ; i++){
				if(canm[i] == ':'){
					col = i;
					break;
				}
			}
			
			int ver_1 = col; //��һ�����ŵ�λ�� 
			for(i = col; ; i++){
				if(canm[i] == '|'){
					ver_1 = i;
					break;
				}
			}
			
			int at = col; //��һ��@��λ�� 
			for(i = col; ; i++){
				if(canm[i] == '@'){
					at = i;
					break;
				}
			}
			
			int be,le; //��beλ��ʼ����leλ����
			be = atoi(canm.substr(col+2, ver_1-col-2).data());
			le = atoi(canm.substr(ver_1+1,at-ver_1-1).data());
			
			int ord; //���С������
			
			temp = canm.substr(at+1,1);
			ord = temp[0] - '0'; //ord������С������
			
			double a,b,c,d;
			
			int zuok = at;//������
			 for(i = at; ; i++){
				if(canm[i] == '('){
					zuok = i;
					break;
				}
			}
			
			int dou = zuok;//����
			 for(i = zuok; ; i++){
				if(canm[i] == ','){
					dou = i;
					break;
				}
			}
			
			int yuok = zuok;//������
			 for(i = zuok; ; i++){
				if(canm[i] == ')'){
					yuok = i;
					break;
				}
			}
			
			int zuof = at;//������
			 for(i = at; ; i++){
				if(canm[i] == '['){
					zuof = i;
					break;
				}
			}
			
			int colf = zuof;//����
			 for(i = zuof; ; i++){
				if(canm[i] == '|'){
					colf = i;
					break;
				}
			}
			
			int yuof = zuof;//�ҷ�����
			 for(i = zuof; ; i++){
				if(canm[i] == ']'){
					yuof = i;
					break;
				}
			}
			
			a = atof(canm.substr(zuok+1,dou-zuok-1).data());
			b = atof(canm.substr(dou+1,yuok-dou-1).data());
			c = atof(canm.substr(zuof+1,colf-zuof-1).data());
			d = atof(canm.substr(colf+1,yuof-colf-1).data());
			
			int mid = (d+c) / 2 + 1; //��ȷֵ(ȡ�м�)
			int low = c - 1;//�ͱ߽�ֵ 
			int high = d + 1;//�߽߱�ֵ
			
			int xm = (mid - b) / a;
			int xl = (low - b) / a;
			int xh = (high - b) / a;   
			
			if(ord == 1){
				mid--;
				xm = (mid - b) / a;
				string xm_2 = chan10_2(xm,1,le);
				int can_2[64];
				for(int i = 0; i < 64; i++)
					can_2[i] = 0;
				for(int i = be; i < be+le && i-be < xm_2.length(); i++){
					can_2[i] = xm_2[i-be]-'0';
				}
				fout << canm << endl;
				fout << "phy:" << mid << endl;
				foutphy << canm << endl;
				foutphy << mid << endl; 
				fout <<'t'<< chan10_16(id_10) 
					<< len_t <<chan2_16(can_2,len_t) << endl << endl;
				foutnr <<'t'<< chan10_16(id_10)
					<< len_t <<chan2_16(can_2,len_t) << "\\r\\n" << endl << endl;
				foutr <<'t'<< chan10_16(id_10)
					<< len_t <<chan2_16(can_2,len_t) << "\\r" << endl << endl;	
				//�������
				//1.Խ�½�
				int xlow = (c-b)/a;
				if(xlow > 0){
					string xl_2 = chan10_2(xl,1,le);
					int can_2l[64];
					for(int i = 0; i < 64; i++)
						can_2l[i] = 0;
					for(int i = be; i < be+le && i-be < xl_2.length(); i++){
						can_2l[i] = xl_2[i-be]-'0';
					}
					fout << "Խ�½����:" << endl;
					fout <<'t'<< chan10_16(id_10) 
					<< len_t <<chan2_16(can_2l,len_t) << endl << endl;
				} 
				
				//2.Խ�Ͻ�
				int xhigh = (d-b)/a;
				if(xhigh < pow(2,le)-1){
					string xh_2 = chan10_2(xh,1,le);
					int can_2h[64];
					for(int i = 0; i < 64; i++)
						can_2h[i] = 0;
					for(int i = be; i < be+le && i-be < xh_2.length(); i++){
						can_2h[i] = xh_2[i-be]-'0';
					}
					fout << "Խ�Ͻ����:" << endl;
					fout <<'t'<< chan10_16(id_10) 
					<< len_t <<chan2_16(can_2h,len_t) << endl << endl;
				}
			}
			
			if(ord == 0){
				string xm_2 = chan10_2(xm,1,le);
				cout << xm << ' ' << xm_2 << endl; 
				int can_2[64];
				for(int i = 0; i < 64; i++)
					can_2[i] = 0;
				zerosort(can_2 ,xm_2,be,le);
				
				for(int i = 0; i < 8; i++){
					for(int j = i * 8 + 7 ; j >= i * 8; j--)
						cout << can_2[j] <<' ';
					cout << endl;
				}
				
				fout << canm << endl;
				fout << "phy:" << mid << endl;
				foutphy << canm << endl;
				foutphy << mid << endl; 
				fout <<'t'<< chan10_16(id_10) 
					<< len_t <<chan2_16(can_2,len_t) << endl << endl;
				foutnr <<'t'<< chan10_16(id_10)
					<< len_t <<chan2_16(can_2,len_t) << "\\r\\n" << endl << endl;	
				foutr <<'t'<< chan10_16(id_10)
					<< len_t <<chan2_16(can_2,len_t) << "\\r" << endl << endl;	
				//�������
				//1.Խ�½�
				int xlow = (c-b)/a;
				if(xlow > 0){
					string xl_2 = chan10_2(xl,1,le);
					int can_2l[64];
					for(int i = 0; i < 64; i++)
						can_2l[i] = 0;
					for(int i = be; i < be+le && i-be < xl_2.length(); i++){
						can_2l[i] = xl_2[i-be]-'0';
					}
					fout << "Խ�½����:" << endl;
					fout <<'t'<< chan10_16(id_10) 
					<< len_t <<chan2_16(can_2l,len_t) << endl << endl;
				} 	
			}
		}
	}
    return 0;
}


//ʮ����ת������ 
string chan10_2(int num,int ord,int le){
	string s="";  
	if(num < 0)
		s+='1';
    for(int a = num; a ;a = a/2)  
    {  
        s=s+(a%2?'1':'0');  
    }
	if(ord == 0){//orderȡ1�������orderȡ0�������� 
		reverse(s.begin(),s.end());
	}
	if(s=="")
		s=s+'0';
	return s;  
}

//������תʮ������
string chan2_16(int * a2, int len_t){
	string s="";
	len_t = len_t * 2;
	char a = '0';
	char b = '0';
	for(int i = 0; i < len_t * 4; i=i+8){
		int sum = a2[i+3]*8 + a2[i+2]*4 + a2[i+1]*2 + a2[i];
		if(sum <= 9)
			a = sum + '0';
		else if(sum > 9)
			a = 'A' + sum - 10;
		sum = 0;
		sum = a2[i+7]*8 + a2[i+6]*4 + a2[i+5]*2 + a2[i+4];
		if(sum <= 9)
			b = sum + '0';
		else if(sum > 9)
			b = 'A' + sum - 10;
		s = s + b;
		s = s + a;
	}
	return s;
}
//ʮ����תʮ������ 
string chan10_16(string a10){
	int n = 0;
	n = atoi(a10.data());
	string s = "";
    if (n == 0)
        s = "0";
        while (n != 0)
        {
            if (n%16 >9 )
                s += n%16 - 10 +'A';
            else
                s += n%16 + '0';
            n = n / 16;
        }

        reverse(s.begin(), s.end());
        
        return s;
} 

void zerosort(int * tar,string x, int be,int le){
	int dev[8]={8,16,24,32,40,48,56,64};
	int first,last;
	int i;
	for(i = 0; i < 8; i++){
		if(be < dev[i])
			break;
	}
	first = i;
	int first_len = be - dev[first] + 8 + 1;
	if(first_len > le)
		first_len = le;
	int red = le - first_len;
	
	cout << be << ' ' << le << endl;
	
	int mid_len = red / 8;
	int last_len = red % 8;
	
	int guid = le - 1; //ָ�� 
	//��һ�� 
	for(int i = 0; i < first_len; i++){
		tar[be-i] = x[le - i - 1] - '0';
		if(tar[be-i] < 0) tar[be-i] = 0;
		guid--;
	}
	
	//�м��� 
	for(int i = 0; i < mid_len ; i++){
		int row = first + 1 + i;
		for(int j = 7; j >= 0; j--){
			tar[row * 8 + j] = x[guid] - '0';
			guid--;
		}
	}
	//���һ��
	for(int i = 0; i < last_len; i++){
		int loc = (first + mid_len + 2)*8-1-i;
		tar[loc] = x[guid] - '0';
		guid--; 
	}
	
	
	return;
	
}

