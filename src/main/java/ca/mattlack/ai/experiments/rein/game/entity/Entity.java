package ca.mattlack.ai.experiments.rein.game.entity;

import ca.mattlack.ai.experiments.rein.game.map.World;
import ca.mattlack.ai.experiments.rein.game.math.Vector2D;

import java.util.UUID;

public abstract class Entity {
    private UUID id = UUID.randomUUID();

    private Vector2D position = new Vector2D(0, 0);
    private Vector2D velocity = new Vector2D(0, 0);

    private final World world;

    public Entity(World world) {
        this.world = world;
    }

    public void spawn() {
        world.addEntity(this);
    }

    public World getWorld() {
        return world;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public UUID getId() {
        return id;
    }

    public void setPosition(Vector2D position) {
        this.position = position;
    }

    public Vector2D getPosition() {
        return position;
    }

    public void setVelocity(Vector2D velocity) {
        this.velocity = velocity;
    }

    public Vector2D getVelocity() {
        return velocity;
    }

    public void remove() {
        world.removeEntity(this);
    }

    public void tick() {
        move(velocity);
        // Reduce velocity by 1%.
        velocity.multiply(0.99);
    }

    public void move(Vector2D movement) {
        //TODO Collision checks.

        // Check if new position is outside of the 128x128 bound.
        if (position.add(movement).getX() > 127 || position.add(movement).getX() < 0 ||
                position.add(movement).getY() > 127 || position.add(movement).getY() < 0) {
            return;
        }

        position = position.add(movement);
    }
}
