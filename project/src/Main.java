import gnu.io.SerialPort;
import serialException.NoSuchPort;
import serialException.NotASerialPort;
import serialException.PortInUse;
import serialException.SerialPortParameterFailure;
import serialPort.SerialTool;


public class Main {

	public static void main(String arg[]) throws SerialPortParameterFailure, NotASerialPort, NoSuchPort, PortInUse{
		System.out.print("begin");
		SerialPort serialPort = SerialTool.openPort("COM12", 115200);
		CANTool tool = new CANTool(serialPort);
		SerialListener listener = new SerialListener(serialPort,tool);
		tool.addListener(listener);
	}
}
