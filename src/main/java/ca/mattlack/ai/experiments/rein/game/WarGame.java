package ca.mattlack.ai.experiments.rein.game;

import ca.mattlack.ai.experiments.rein.game.entity.EntityPlayer;
import ca.mattlack.ai.experiments.rein.game.map.World;
import ca.mattlack.ai.experiments.rein.game.math.Vector2D;

import java.util.concurrent.ThreadLocalRandom;

// 2 players.
public class WarGame {
    public final World world = new World();

    public EntityPlayer player1, player2;

    public Vector2D flagPos1 = new Vector2D(world.getMapSize() / 2d + ThreadLocalRandom.current().nextDouble(-64, 64), ThreadLocalRandom.current().nextDouble(2, 45));
    public Vector2D flagPos2 = new Vector2D(world.getMapSize() / 2d + ThreadLocalRandom.current().nextDouble(-64, 64), 128 - ThreadLocalRandom.current().nextDouble(2, 45));

    public void setup() {
        player1 = new EntityPlayer(world);
        player2 = new EntityPlayer(world);

        // Change: set flag positions to completely random positions on the map.
        flagPos1 = new Vector2D(ThreadLocalRandom.current().nextDouble(0, world.getMapSize()), ThreadLocalRandom.current().nextDouble(0, world.getMapSize()));
        flagPos2 = new Vector2D(ThreadLocalRandom.current().nextDouble(0, world.getMapSize()), ThreadLocalRandom.current().nextDouble(0, world.getMapSize()));

        player1.setPosition(flagPos1);
        player2.setPosition(flagPos2);

        player1.spawn();
        player2.spawn();

        world.setBlock((int) flagPos1.getX(), (int) flagPos1.getY(), 1);
        world.setBlock((int) flagPos2.getX(), (int) flagPos2.getY(), 1);
    }

    public void movePlayer(EntityPlayer player, double x, double y) {
        player.move(new Vector2D(x, y));
    }

    public void breakBlock(int x, int y) {
        world.setBlock(x, y, 0);
    }

    public World getWorld() {
        return world;
    }

    public int getWinner() {
        int flagX = (int) flagPos1.getX();
        int flagY = (int) flagPos1.getY();

        int block = world.getBlock(flagX, flagY);

        boolean flag1Broken = block == 0;

        flagX = (int) flagPos2.getX();
        flagY = (int) flagPos2.getY();

        block = world.getBlock(flagX, flagY);

        boolean flag2Broken = block == 0;

        return flag1Broken && flag2Broken ? 2 : (flag1Broken ? 1 : (flag2Broken ? 0 : -1));
    }

    public WarGameState getCurrentGameState() {
        int[] player1Env = new int[49]; // 7x7 around the player.
        int[] player2Env = new int[49];
        for (int i = -3; i < 4; i++) {
            for (int j = -3; j < 4; j++) {
                player1Env[(i + 3) * 7 + j + 3] = world.getBlock((int) player1.getPosition().getX() + i, (int) player1.getPosition().getY() + j);
                player2Env[(i + 3) * 7 + j + 3] = world.getBlock((int) player2.getPosition().getX() + i, (int) player2.getPosition().getY() + j);
            }
        }
        return new WarGameState(player1.getPosition(), player2.getPosition(), player1Env, player2Env, flagPos1, flagPos2);
    }


}
