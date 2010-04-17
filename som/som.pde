final int BG_COLOR = 255;
final int LEFT_MARGIN = 50;
final int TOP_MARGIN = 50;

void draw_hexagon(int x, int y, int circumdiameter, boolean hover) {
	float R = float(circumdiameter) / 2.0;
	float r = R * sqrt(3.0) / 2.0;
	stroke(100);
	if (hover) {
		fill(100);
	} 
	else {
		fill(255);
	}
	beginShape();
	vertex(x, y + r);
	vertex(x + R / 2.0, y + 2 * r);
	vertex(x + (3/2.0) * R, y + 2 * r);
	vertex(x + 2 * R, y + r);
	vertex(x + (3/2.0) * R, y);
	vertex(x + R / 2.0, y);
	endShape(CLOSE);
}

void draw_hexagonal_grid(int width, int height, int circumdiameter) {
	double cr = (double)circumdiameter;
	double ir = sqrt(3.0) * cr / 2.0;
	for (int i=0; i < width; i++) {
		for (int j=0; j < height; j ++) {
			double offset = ir;
			if (i % 2 == 0) {
				offset = 0.0;
			}
			double x = (double)LEFT_MARGIN + i * (3.0 / 2.0) * cr;
			double y = (double)TOP_MARGIN + j * 2.0 * ir + offset;
			double next_x = (double)LEFT_MARGIN + (i + 1) * (3.0 / 2.0) * cr;
			double next_y = (double)TOP_MARGIN + (j+1) * 2 * ir + offset;
			boolean fill_ = false;
			if (mouseX > x && mouseY > y && mouseX < next_x && mouseY < next_y ) {
				fill_ = true;
				fill(0);
				//text(i + ", " + j, 100, 800);
			}
			draw_hexagon((int)x, (int)y, (int)(cr*2), fill_);
		}
	}
}

void setup() {
	smooth();
	size(1000,1000);
}

void draw() {
	background(BG_COLOR);
	draw_hexagonal_grid(20,10,20);
}