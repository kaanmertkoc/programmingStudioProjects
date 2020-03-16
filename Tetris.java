import java.awt.Color;
import java.awt.Font;

class Box {
	
	 Color color;
	 int number;
	 double halfLength;
	 double x;
	 double y;
	
	Box(Color color, int number, double halfLength, double x, double y) {
		this.color = color;
		this.number = number;
		this.halfLength = halfLength;
		this.x = x;
		this.y = y;
		
	} 
}
public class Tetris {

	public static void main(String[] args) {
		drawMap();
		Box box = new Box(Color.yellow, 2, 0.2, 0.0, 0.0);
		createBox(box);
		createBox(new Box(Color.yellow, 2, 0.2, 0.0, 1.8));

	}
	// First version of the map maybe this can be implemented by input number later
	public static void drawMap() {
		StdDraw.setCanvasSize(500, 500);

		StdDraw.setXscale(0.0, 2.0);
		StdDraw.setYscale(0.0, 2.0);
		StdDraw.setPenColor(187, 173, 160);

		double x = 0.0;
		double y = 0.0;

		for(double i = x; i <= 1.50; i = i + 0.2) {
			for(double j = y; j <= 2.0; j = j + 0.2) {

				StdDraw.filledSquare(i, j, 0.2);
			}
		}
		StdDraw.setPenColor(Color.gray);
		for(double i = x; i <= 1.50; i = i + 0.2) {
			for(double j = y; j <= 2.0; j = j + 0.2) {
				StdDraw.square(i, j, 0.2);

			}
		}
		for(double i = 1.60; i <= 2.0; i = i + 0.2) {
			for(double j = y; j <= 2.0; j = j + 0.2) {
				StdDraw.filledSquare(i, j, 0.2);
			}
		}
		Font font = new Font("Arial", Font.BOLD, 40);
		StdDraw.setFont(font);
		StdDraw.setPenColor(Color.white);
		StdDraw.text(1.70, 1.80, "Score");
		
	
	}
	public static void createBox(Box box) {
			Font font = new Font("Arial", Font.BOLD, 30);
			StdDraw.setFont(font);
			double textAxis = (box.halfLength * 2) * (Math.sqrt(2) / 2);
			
			StdDraw.setPenColor(box.color);
			StdDraw.filledSquare(box.x + (box.halfLength / 2), box.y + (box.halfLength / 2), box.halfLength);		
			StdDraw.setPenColor(Color.gray);
			StdDraw.text(box.x + (box.halfLength / 2), box.y + (box.halfLength / 2), Integer.toString(box.number));
			StdDraw.setPenColor(Color.gray);
			StdDraw.square(box.x, box.y, box.halfLength);
	}
	

}
