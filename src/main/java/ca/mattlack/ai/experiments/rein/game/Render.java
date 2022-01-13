package ca.mattlack.ai.experiments.rein.game;

import ca.mattlack.ai.experiments.rein.game.math.Vector2D;

import java.util.List;

import javax.swing.*;
import java.awt.*;
import java.awt.image.VolatileImage;
import java.security.SecureRandom;
import java.util.ArrayList;

public class Render {
    public final JFrame frame;

    private volatile int mouseX = -1;
    private volatile int mouseY = -1;

    public Render() {
        this.frame = new JFrame("Reinforcement Learning");
        frame.setSize(128 * 8, 128 * 8);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.createBufferStrategy(2);

        // Create a mouse listener.
        frame.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent e) {
                mouseX = e.getX();
                mouseY = e.getY();
            }
        });
    }

    boolean b = false;

    public void renderFrame(WarGame game, Vector2D... highlightedSquares) {
        // Each square will be 8 pixels wide. There are 128 squares.

        Graphics g = frame.getBufferStrategy().getDrawGraphics();

        if (mouseX != -1) {
            Vector2D pos = new Vector2D(mouseX / 8d, mouseY / 8d);
            game.flagPos1 = pos;
            game.getWorld().setBlock((int) pos.getX(), (int) pos.getY(), 1);
        }

        g.setColor(Color.DARK_GRAY.darker());
        g.fillRect(0, 0, frame.getWidth(), frame.getHeight());

        // Divide using lines.
        g.setColor(Color.DARK_GRAY);
        for (int i = 0; i < 128; i++) {
            g.drawLine(i * 8, 0, i * 8, 128 * 8);
        }
        for (int i = 0; i < 128; i++) {
            g.drawLine(0, i * 8, 128 * 8, i * 8);
        }

        Vector2D flag1 = game.flagPos1;
        g.setColor(Color.RED);
        g.fillOval((int) (flag1.getX() * 8) - 4,
                (int) (flag1.getY() * 8) - 4,
                8,
                8);

        Vector2D playerPos = game.player1.getPosition();
        g.setColor(Color.BLUE);
        g.fillOval((int) (playerPos.getX() * 8) - 4,
                (int) (playerPos.getY() * 8) - 4,
                8,
                8);

        Vector2D flag2 = game.flagPos2;
        g.setColor(Color.PINK);
        g.fillOval((int) (flag2.getX() * 8) - 4,
                (int) (flag2.getY() * 8) - 4,
                8,
                8);

        Vector2D playerPos2 = game.player2.getPosition();
        g.setColor(Color.BLUE);
        g.fillOval((int) (playerPos2.getX() * 8) - 4,
                (int) (playerPos2.getY() * 8) - 4,
                8,
                8);

        // Highlight squares.
        g.setColor(Color.GREEN);
        for (Vector2D v : highlightedSquares) {
            g.fillRect((int) (v.getX() * 8), (int) (v.getY() * 8), 8, 8);
        }


        frame.getBufferStrategy().show();
        g.dispose();
    }

    public static void main(String[] args) {

        List<Integer> list = new ArrayList<>();

        for (int j = 0; j < 1000; j++) {
            int count = 0;
            for (int i = 0; i < 10000; i++) {
                if (Math.random() < 0.005) {
                    count++;
                }
            }
            list.add(count);
        }

        // Average the numbers in the list.
        int sum = 0;
        for (int i : list) {
            sum += i;
        }
        double average = sum / (double) list.size();
        System.out.println("Average: " + average);

        // Find the standard deviation.
        double sumOfSquares = 0;
        for (int i : list) {
            sumOfSquares += Math.pow(i - average, 2);
        }
        double standardDeviation = Math.sqrt(sumOfSquares / (double) list.size());
        System.out.println("Standard Deviation: " + standardDeviation);
    }

}
