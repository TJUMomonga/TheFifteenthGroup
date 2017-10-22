import serialException.TooManyListeners;
import serialPort.SerialTool;
import gnu.io.SerialPort;


public class CANTool {
	private SerialPort serialPort;
	private int state;
	private int speed;
	
	public CANTool(SerialPort serialPort)
	{
		this.serialPort = serialPort;
		state = 0;
		speed = 10;
	}
	
	public void addListener(SerialListener listener)
	{
		try {
			SerialTool.addListener(serialPort, listener);
		} catch (TooManyListeners e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void readCommand(String command)
	{
		if(command == null || command.length() == 0)
			returnTheInfo(0,"");
		char type = command.charAt(0);
		command = command.substring(0,command.length()-1);
		if(type=='V' && command.length() == 1)//接受到“V/r”,版本信息1
		{
			returnTheInfo(1,"SV2.5-HV2.0");
		}
		else if(type=='O' && command.charAt(1) == '1' && command.length() == 2)//接受到“O1/r”，open打开2
		{
			open();
		}
		else if(type=='C' && command.length() == 1)//“C/r”关闭close4
		{
			close();
		}
		else if(type=='S' && command.length() == 2)//设置CAN总线的通信速率3
		{
			changeSpeed(command.charAt(1));//数字代表通信速率
		}
		else if(type=='T')//拓展帧5
		{
			try {
				sendExtendedFrame(command);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		else if(type=='t')//标准帧6
		{
			try {
				sendStandardFrame(command);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		else {
			returnTheInfo(0,"");
		}
		
	}
	
	public void sendStandardFrame(String command) throws Exception, Exception 
	{
		if(state == 0)
		{
			returnTheInfo(0,"");
			return;
		}
		int templen = command.length();
		for(int i=1;i<templen;i++)
		{
			char tempchar = command.charAt(i);
			if(!((tempchar>='0'&&tempchar<='9')||(tempchar>='A'&&tempchar<='F')))
			{
				returnTheInfo(0,"");
				return;
			}
		}
		if(templen<=4)//t+ID最小4位
		{
			returnTheInfo(0,"");
			return;
		}
		//substring长度4-1=3不包括第四位
		String idString = command.substring(1, 4);//ID
		String lenString = command.substring(4, 5);//长度byte,2个16进制，8bit
		int id = Integer.parseInt(idString, 16);
		int len = Integer.parseInt(lenString, 16);
		if(len<=0||len>8||templen!=9+len*2)//最长64bit，9=TYPE1+ID3+DLC1+TIME4
		{
			returnTheInfo(0,"");
			return;
		}
		String data_16 = command.substring(5, 5+len*2);
		String timeString = command.substring(5+len*2,9+len*2);
		String data_2 = "";//将16进制转化为2进制
		for(int i=0;i<len*2;i++)
		{
			data_2 = data_2 + Integer.toBinaryString(Integer.parseInt(data_16.substring(i,i+1), 16));
		}
		int time = Integer.parseInt(timeString, 16);
		if(CheckFormat.check(id,Long.parseLong(data_16,16)))
		{
			returnTheInfo(1,"");
			if(time == 0)//代表发送一次
			{
				System.out.println(idString+lenString+data_16);
			}
			else
			{
				Thread th = new Thread();
				for(int i=0;i<100;i++)
				{
					System.out.println(idString+lenString+data_16);
					try {
						th.sleep(time);//以time周期发送
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				th.stop();
			}
		}
		else
		{
			returnTheInfo(0,"");
		}
		
	}
	
	//发送拓展帧
	public void sendExtendedFrame(String command) throws Exception, Exception 
	{
		if(state == 0)
		{
			returnTheInfo(0,"");
			return;
		}
		int templen = command.length();
		for(int i=1;i<templen;i++)
		{
			char tempchar = command.charAt(i);
			if(!((tempchar>='0'&&tempchar<='9')||(tempchar>='A'&&tempchar<='F')))
			{
				returnTheInfo(0,"");
				return;
			}
		}
		if(templen<=9)//T+ID=9
		{
			returnTheInfo(0,"");
			return;
		}
		String idString = command.substring(1, 9);
		
		String lenString = command.substring(9, 10);
		
		int id = Integer.parseInt(idString, 16);
		int len = Integer.parseInt(lenString, 16);
		if(len<=0||len>8||templen!=14+len*2)
		{
			returnTheInfo(0,"");
			return;
		}
		String data_16 = command.substring(10, 10+len*2);
		String timeString = command.substring(10+len*2,14+len*2);
		String data_2 = "";
		for(int i=0;i<len*2;i++)
		{
			data_2 = data_2 + Integer.toBinaryString(Integer.parseInt(data_16.substring(i,i+1), 16));
		}
		int time = Integer.parseInt(timeString, 16);
		if(CheckFormat.check(id,Long.parseLong(data_16,16)))
		{
			returnTheInfo(1,"");
			if(time == 0)
			{
				System.out.println(idString+lenString+data_16);
			}
			else
			{
				Thread th = new Thread();
				for(int i=0;i<100;i++)
				{
					System.out.println(idString+lenString+data_16);
					try {
						th.sleep(time);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				th.stop();
			}
		}
		else
		{
			returnTheInfo(0,"");
		}
	}
	
	public void changeSpeed(char c) 
	{
		if(state == 0)
		{
			int level = (int)(c-'0');
			if(level<0 || level >8)//S0-S8 3
				returnTheInfo(0,"");
			else
			{
				int num[]={10,20,50,100,125,250,500,800,1000};
				speed=num[level];
				System.out.println(speed);
				returnTheInfo(1,"");
			}
			
		}
		else
		{
			returnTheInfo(0,"");
		}
		
	}
	
}
