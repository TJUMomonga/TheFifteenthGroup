#include <fstream>
#include <cstdio>
#include <cstring>

using namespace std;

int main(){
    ifstream fin("data.in");
    ofstream fout("data.txt");
    
    string canm;
    int num = 1;
    
    while(getline(fin,canm)){
    	string mes = canm.substr(0,3);
    	if(mes == "BO_"){
    		
		}
	}
    
    return 0;
}
