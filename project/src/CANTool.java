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
		if(type=='V' && command.length() == 1)//���ܵ���V/r��,�汾��Ϣ1
		{
			returnTheInfo(1,"SV2.5-HV2.0");
		}
		else if(type=='O' && command.charAt(1) == '1' && command.length() == 2)//���ܵ���O1/r����open��2
		{
			open();
		}
		else if(type=='C' && command.length() == 1)//��C/r���ر�close4
		{
			close();
		}
		else if(type=='S' && command.length() == 2)//����CAN���ߵ�ͨ������3
		{
			changeSpeed(command.charAt(1));//���ִ���ͨ������
		}
		else if(type=='T')//��չ֡5
		{
			try {
				sendExtendedFrame(command);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		else if(type=='t')//��׼֡6
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
		if(templen<=4)//t+ID��С4λ
		{
			returnTheInfo(0,"");
			return;
		}
		//substring����4-1=3����������λ
		String idString = command.substring(1, 4);//ID
		String lenString = command.substring(4, 5);//����byte,2��16���ƣ�8bit
		int id = Integer.parseInt(idString, 16);
		int len = Integer.parseInt(lenString, 16);
		if(len<=0||len>8||templen!=9+len*2)//�64bit
		{
			returnTheInfo(0,"");
			return;
		}
		String data_16 = command.substring(5, 5+len*2);
		String timeString = command.substring(5+len*2,9+len*2);
		String data_2 = "";
		for(int i=0;i<len*2;i++)
		{
			data_2 = data_2 + Integer.toBinaryString(Integer.parseInt(data_16.substring(i,i+1), 16));
		}
		int time = Integer.parseInt(timeString, 16);
		if(CheckFormat.check(id,Long.parseUnsignedLong(data_16,16)))
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
	
}