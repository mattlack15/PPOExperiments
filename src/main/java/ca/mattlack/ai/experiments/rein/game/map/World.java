package ca.mattlack.ai.experiments.rein.game.map;

import ca.mattlack.ai.experiments.rein.game.entity.Entity;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public class World {
    private Map<UUID, Entity> entityMap = new ConcurrentHashMap<>();

    private final int[] map = new int[128 * 128];

    public void addEntity(Entity entity) {
        entityMap.put(entity.getId(), entity);
    }

    public Map<UUID, Entity> getEntityMap() {
        return entityMap;
    }

    public void tick() {
        // Tick all entities.
        entityMap.values().forEach(Entity::tick);
    }

    public void removeEntity(Entity entity) {
        entityMap.remove(entity.getId());
    }

    public int getBlock(int x, int y) {
        if (x < 0 || y < 0) return 0;
        if (x > 127 || y > 127) return 0;
        return map[x << 7 | y];
    }

    public void setBlock(int x, int y, int block) {
        if (x < 0 || y < 0) return;
        if (x > 127 || y > 127) return;
        map[x << 7 | y] = block;
    }

    public int[] getMap() {
        return map;
    }

    public int getMapSize() {
        return 128;
    }
}
