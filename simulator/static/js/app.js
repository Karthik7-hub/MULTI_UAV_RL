/**
 * AEROPLAN 3D — UAV Path-Planning & Policy Inspector
 * Scientific WebGL 3D CAD Visualization Engine & Decision Telemetry Controller
 */

// Application State
const state = {
    world: {
        bounds: { x_min: 0, x_max: 200, y_min: 0, y_max: 200, z_min: 0, z_max: 100 },
        obstacles: [],
        source: [20, 20, 10],
        target: [180, 180, 10],
        margin: 2.0,
        gridRes: 5.0
    },
    activePlanner: 'hybrid_astar_mappo',
    compareMode: false,
    comparisonResults: null,
    visiblePlanners: {
        hybrid_astar_mappo: true,
        astar: true,
        mappo_residual: true,
        mappo_pure: true
    },
    plannedTrajectory: null,
    currentStepIdx: 0,
    isPlaying: false,
    playSpeed: 1.0,
    layers: {
        showPath: true,
        showExplored: true,
        showHalos: false
    }
};

// Preset Scenarios
const PRESETS = {
    benchmark: [
        { id: 0, type: 'cylinder', cx: 142.5, cy: 100.0, radius: 10.0, height: 80.0, enabled: true },
        { id: 1, type: 'cylinder', cx: 100.0, cy: 142.5, radius: 10.0, height: 85.0, enabled: true },
        { id: 2, type: 'cylinder', cx: 57.5, cy: 100.0, radius: 10.0, height: 80.0, enabled: true },
        { id: 3, type: 'cylinder', cx: 100.0, cy: 57.5, radius: 10.0, height: 75.0, enabled: true }
    ],
    metropolis: [
        { id: 0, type: 'cylinder', cx: 140.0, cy: 100.0, radius: 10.0, height: 85.0, enabled: true },
        { id: 1, type: 'cylinder', cx: 100.0, cy: 140.0, radius: 10.0, height: 90.0, enabled: true },
        { id: 2, type: 'cylinder', cx: 60.0, cy: 100.0, radius: 10.0, height: 80.0, enabled: true },
        { id: 3, type: 'cylinder', cx: 100.0, cy: 60.0, radius: 10.0, height: 75.0, enabled: true },
        { id: 4, type: 'cylinder', cx: 135.0, cy: 135.0, radius: 11.0, height: 92.0, enabled: true },
        { id: 5, type: 'cylinder', cx: 65.0, cy: 135.0, radius: 12.0, height: 88.0, enabled: true },
        { id: 6, type: 'cylinder', cx: 65.0, cy: 65.0, radius: 11.0, height: 82.0, enabled: true },
        { id: 7, type: 'cylinder', cx: 135.0, cy: 65.0, radius: 12.0, height: 86.0, enabled: true },
        { id: 8, type: 'cylinder', cx: 165.0, cy: 145.0, radius: 10.0, height: 70.0, enabled: true },
        { id: 9, type: 'cylinder', cx: 35.0, cy: 145.0, radius: 9.0, height: 75.0, enabled: true },
        { id: 10, type: 'cylinder', cx: 35.0, cy: 55.0, radius: 10.0, height: 68.0, enabled: true },
        { id: 11, type: 'cylinder', cx: 165.0, cy: 55.0, radius: 9.0, height: 72.0, enabled: true },
        { id: 12, type: 'cylinder', cx: 145.0, cy: 165.0, radius: 8.0, height: 65.0, enabled: true },
        { id: 13, type: 'cylinder', cx: 55.0, cy: 165.0, radius: 8.0, height: 78.0, enabled: true },
        { id: 14, type: 'cylinder', cx: 55.0, cy: 35.0, radius: 9.0, height: 60.0, enabled: true },
        { id: 15, type: 'cylinder', cx: 145.0, cy: 35.0, radius: 8.0, height: 70.0, enabled: true },
        { id: 16, type: 'cylinder', cx: 175.0, cy: 100.0, radius: 8.0, height: 55.0, enabled: true },
        { id: 17, type: 'cylinder', cx: 100.0, cy: 175.0, radius: 8.0, height: 60.0, enabled: true },
        { id: 18, type: 'cylinder', cx: 25.0, cy: 100.0, radius: 8.0, height: 50.0, enabled: true },
        { id: 19, type: 'cylinder', cx: 100.0, cy: 25.0, radius: 8.0, height: 55.0, enabled: true }
    ],
    chokepoint: [
        { id: 0, type: 'cylinder', cx: 142.5, cy: 112.0, radius: 7.0, height: 90.0, enabled: true },
        { id: 1, type: 'cylinder', cx: 142.5, cy: 88.0, radius: 7.0, height: 90.0, enabled: true },
        { id: 2, type: 'cylinder', cx: 57.5, cy: 112.0, radius: 7.0, height: 90.0, enabled: true },
        { id: 3, type: 'cylinder', cx: 57.5, cy: 88.0, radius: 7.0, height: 90.0, enabled: true },
        { id: 4, type: 'cylinder', cx: 112.0, cy: 142.5, radius: 7.0, height: 90.0, enabled: true },
        { id: 5, type: 'cylinder', cx: 88.0, cy: 142.5, radius: 7.0, height: 90.0, enabled: true },
        { id: 6, type: 'cylinder', cx: 112.0, cy: 57.5, radius: 7.0, height: 90.0, enabled: true },
        { id: 7, type: 'cylinder', cx: 88.0, cy: 57.5, radius: 7.0, height: 90.0, enabled: true }
    ],
    canyon: [
        { id: 0, type: 'cylinder', cx: 100, cy: 60, radius: 24, height: 90, enabled: true },
        { id: 1, type: 'cylinder', cx: 100, cy: 140, radius: 24, height: 90, enabled: true },
        { id: 2, type: 'cylinder', cx: 80, cy: 100, radius: 12, height: 80, enabled: true }
    ],
    maze: [
        { id: 0, type: 'cylinder', cx: 50, cy: 60, radius: 14, height: 80, enabled: true },
        { id: 1, type: 'cylinder', cx: 90, cy: 40, radius: 15, height: 80, enabled: true },
        { id: 2, type: 'cylinder', cx: 140, cy: 60, radius: 16, height: 80, enabled: true },
        { id: 3, type: 'cylinder', cx: 70, cy: 110, radius: 18, height: 80, enabled: true },
        { id: 4, type: 'cylinder', cx: 120, cy: 110, radius: 15, height: 80, enabled: true },
        { id: 5, type: 'cylinder', cx: 160, cy: 120, radius: 14, height: 80, enabled: true },
        { id: 6, type: 'cylinder', cx: 60, cy: 160, radius: 16, height: 80, enabled: true },
        { id: 7, type: 'cylinder', cx: 110, cy: 170, radius: 15, height: 80, enabled: true }
    ],
    spheres: [
        { id: 0, type: 'sphere', cx: 65, cy: 65, cz: 35, radius: 18, enabled: true },
        { id: 1, type: 'cylinder', cx: 120, cy: 75, radius: 16, height: 85, enabled: true },
        { id: 2, type: 'sphere', cx: 95, cy: 115, cz: 40, radius: 20, enabled: true },
        { id: 3, type: 'cylinder', cx: 135, cy: 135, radius: 15, height: 75, enabled: true },
        { id: 4, type: 'sphere', cx: 155, cy: 90, cz: 50, radius: 16, enabled: true }
    ],
    empty: []
};

// Three.js Components
let scene, camera, renderer, controls;
let droneMesh, rotors = [];
let obstacleMeshes = [];
let haloMeshes = [];
let pathMesh = null;
let multiPathMeshes = {};
let exploredPointCloud = null;
let sourceBeacon = null, targetBeacon = null;
let labelGroup = null;
let cageGroup = null;

const ALGO_COLORS = {
    'hybrid_astar_mappo': 0x1D4ED8,  // Royal Blue
    'astar':              0xDC2626,  // Crimson Red
    'rrt_star':           0x16A34A,  // Forest Green
    'sapf':               0xD97706,  // Amber
    'mappo_residual':     0xEA580C,  // Deep Orange
    'mappo_pure':         0x8B5CF6,  // Vibrant Violet
    'ippo_residual':      0x0D9488,  // Dark Teal
    'ippo_pure':          0x06B6D4,  // Cyan
    'maddpg_residual':    0xEC4899,  // Pink
    'maddpg_pure':        0xF43F5E,  // Rose
    'cppo':               0x6366F1   // Indigo
};

// Coordinate Mapping: Simulator (X, Y, Z_altitude) -> Three.js (X, Z_altitude_up, Y_depth)
function simToThree(x, y, z) {
    return new THREE.Vector3(x, z, y);
}

// -----------------------------------------------------------------------------
// TEXT SPRITE UTILITY (Clean Vector-Style 3D Labels)
// -----------------------------------------------------------------------------
function makeTextSprite(text, options = {}) {
    const fontSize = options.fontSize || 32;
    const fontColor = options.fontColor || '#1e293b';
    const fontWeight = options.fontWeight || '600';
    const scale = options.scale || [12, 6, 1];

    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.font = `${fontWeight} ${fontSize}px Inter, -apple-system, BlinkMacSystemFont, sans-serif`;
    ctx.fillStyle = fontColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 64);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(scale[0], scale[1], scale[2]);
    return sprite;
}

// -----------------------------------------------------------------------------
// INITIALIZATION
// -----------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', async () => {
    initThreeScene();
    setupEventListeners();
    await fetchWorldConfig();
    updateObstacleListUI();
    renderWorldEntities();
    animateScene();

    // Auto-run baseline planning on startup
    runPlanning();
});

// -----------------------------------------------------------------------------
// THREE.JS SCENE SETUP (Light Minimal Engineering CAD Aesthetic)
// -----------------------------------------------------------------------------
function initThreeScene() {
    const container = document.getElementById('canvas-container');

    // 1. Scene & Crisp Pure White Background
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    // 2. Camera Setup (Isometric CAD view matching reference)
    const aspect = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.5, 3000);
    camera.position.set(-90, 200, -90);

    // 3. Renderer with high sharpness
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 4. Orbit Controls (Engineering Orbit & Pan)
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(100, 35, 100);
    controls.maxPolarAngle = Math.PI / 2 - 0.01; // Avoid going beneath ground plane

    // 5. Scientific CAD Studio Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.82);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.45);
    dirLight1.position.set(180, 280, 160);
    dirLight1.castShadow = true;
    dirLight1.shadow.mapSize.width = 2048;
    dirLight1.shadow.mapSize.height = 2048;
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xf1f5f9, 0.25);
    dirLight2.position.set(-150, 200, -100);
    scene.add(dirLight2);

    // 6. 3D Airspace Cage & Coordinate Tick Frame
    createAirspaceCage();

    // 7. Quadrotor UAV Model
    droneMesh = createDroneMesh();
    scene.add(droneMesh);

    window.addEventListener('resize', onWindowResize);
}

// -----------------------------------------------------------------------------
// 3D AIRSPACE CAGE, FLOOR GRID & AXIS TICK LABELS
// -----------------------------------------------------------------------------
function createAirspaceCage() {
    if (cageGroup) scene.remove(cageGroup);
    cageGroup = new THREE.Group();

    const b = state.world.bounds;
    const sizeX = b.x_max - b.x_min;
    const sizeY = b.y_max - b.y_min;
    const sizeZ = b.z_max - b.z_min; // Altitude range (0 to 100)

    // Floor Grid (Sim X -> Three X, Sim Y -> Three Z)
    const gridHelper = new THREE.GridHelper(sizeX, 20, 0xd1d5db, 0xf1f5f9);
    gridHelper.position.set(sizeX / 2, 0, sizeY / 2);
    cageGroup.add(gridHelper);

    // Bounding Box Frame (12 thin gray CAD lines)
    const boxMat = new THREE.LineBasicMaterial({ color: 0x94a3b8, linewidth: 1, transparent: true, opacity: 0.85 });
    const points = [
        // Bottom perimeter
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(sizeX, 0, 0),
        new THREE.Vector3(sizeX, 0, 0), new THREE.Vector3(sizeX, 0, sizeY),
        new THREE.Vector3(sizeX, 0, sizeY), new THREE.Vector3(0, 0, sizeY),
        new THREE.Vector3(0, 0, sizeY), new THREE.Vector3(0, 0, 0),
        // Top perimeter
        new THREE.Vector3(0, sizeZ, 0), new THREE.Vector3(sizeX, sizeZ, 0),
        new THREE.Vector3(sizeX, sizeZ, 0), new THREE.Vector3(sizeX, sizeZ, sizeY),
        new THREE.Vector3(sizeX, sizeZ, sizeY), new THREE.Vector3(0, sizeZ, sizeY),
        new THREE.Vector3(0, sizeZ, sizeY), new THREE.Vector3(0, sizeZ, 0),
        // 4 Vertical Pillars
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, sizeZ, 0),
        new THREE.Vector3(sizeX, 0, 0), new THREE.Vector3(sizeX, sizeZ, 0),
        new THREE.Vector3(sizeX, 0, sizeY), new THREE.Vector3(sizeX, sizeZ, sizeY),
        new THREE.Vector3(0, 0, sizeY), new THREE.Vector3(0, sizeZ, sizeY)
    ];

    const boxGeom = new THREE.BufferGeometry().setFromPoints(points);
    const boxLines = new THREE.LineSegments(boxGeom, boxMat);
    cageGroup.add(boxLines);

    // Axis Ticks and Labels Matching Reference Image
    const tickMat = new THREE.LineBasicMaterial({ color: 0x94a3b8 });

    // X Axis Ticks (0, 50, 100, 150, 200)
    for (let x = 0; x <= sizeX; x += 50) {
        // Tick mark along floor edge (Z = sizeY)
        const tickPts = [new THREE.Vector3(x, 0, sizeY), new THREE.Vector3(x, 0, sizeY + 3.0)];
        const tickGeom = new THREE.BufferGeometry().setFromPoints(tickPts);
        cageGroup.add(new THREE.Line(tickGeom, tickMat));

        const lbl = makeTextSprite(`${x}`, { fontSize: 26, fontColor: '#64748b', scale: [10, 5, 1] });
        lbl.position.set(x, 0, sizeY + 8.0);
        cageGroup.add(lbl);
    }
    const xLabel = makeTextSprite('X (m)', { fontSize: 30, fontColor: '#1e293b', fontWeight: '700', scale: [16, 8, 1] });
    xLabel.position.set(sizeX / 2, 0, sizeY + 16.0);
    cageGroup.add(xLabel);

    // Y Axis Ticks (0, 50, 100, 150, 200)
    for (let y = 0; y <= sizeY; y += 50) {
        const tickPts = [new THREE.Vector3(sizeX, 0, y), new THREE.Vector3(sizeX + 3.0, 0, y)];
        const tickGeom = new THREE.BufferGeometry().setFromPoints(tickPts);
        cageGroup.add(new THREE.Line(tickGeom, tickMat));

        const lbl = makeTextSprite(`${y}`, { fontSize: 26, fontColor: '#64748b', scale: [10, 5, 1] });
        lbl.position.set(sizeX + 8.0, 0, y);
        cageGroup.add(lbl);
    }
    const yLabel = makeTextSprite('Y (m)', { fontSize: 30, fontColor: '#1e293b', fontWeight: '700', scale: [16, 8, 1] });
    yLabel.position.set(sizeX + 16.0, 0, sizeY / 2);
    cageGroup.add(yLabel);

    // Z Axis Ticks (0, 50, 100) along pillar at (0, 0, 0)
    for (let z = 0; z <= sizeZ; z += 50) {
        const tickPts = [new THREE.Vector3(0, z, 0), new THREE.Vector3(-3.0, z, 0)];
        const tickGeom = new THREE.BufferGeometry().setFromPoints(tickPts);
        cageGroup.add(new THREE.Line(tickGeom, tickMat));

        const lbl = makeTextSprite(`${z}`, { fontSize: 26, fontColor: '#64748b', scale: [10, 5, 1] });
        lbl.position.set(-8.0, z, 0);
        cageGroup.add(lbl);
    }
    const zLabel = makeTextSprite('Z (m)', { fontSize: 30, fontColor: '#1e293b', fontWeight: '700', scale: [16, 8, 1] });
    zLabel.position.set(-16.0, sizeZ / 2, 0);
    cageGroup.add(zLabel);

    scene.add(cageGroup);
}

// -----------------------------------------------------------------------------
// UAV DRONE MESH (Minimalist Engineering Precision Drone)
// -----------------------------------------------------------------------------
function createDroneMesh() {
    const drone = new THREE.Group();

    // Central chassis
    const bodyGeom = new THREE.CylinderGeometry(2.2, 2.4, 0.9, 16);
    const bodyMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        roughness: 0.3,
        metalness: 0.6
    });
    const body = new THREE.Mesh(bodyGeom, bodyMat);
    drone.add(body);

    // Red forward heading nose
    const noseGeom = new THREE.ConeGeometry(0.7, 1.8, 8);
    const noseMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const nose = new THREE.Mesh(noseGeom, noseMat);
    nose.rotation.x = Math.PI / 2;
    nose.position.set(0, 0, 2.8);
    drone.add(nose);

    // 4 Arms & Rotors
    const armMat = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.4 });
    const rotorMat = new THREE.MeshBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.7 });
    rotors = [];

    const armOffsets = [[2.8, 2.8], [-2.8, 2.8], [-2.8, -2.8], [2.8, -2.8]];
    armOffsets.forEach(([ax, az]) => {
        const armGeom = new THREE.CylinderGeometry(0.2, 0.2, 3.8);
        const arm = new THREE.Mesh(armGeom, armMat);
        arm.position.set(ax / 2, 0, az / 2);
        arm.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(ax, 0, az).normalize());
        drone.add(arm);

        const rotorGeom = new THREE.CylinderGeometry(1.4, 1.4, 0.05, 12);
        const rotor = new THREE.Mesh(rotorGeom, rotorMat);
        rotor.position.set(ax, 0.45, az);
        drone.add(rotor);
        rotors.push(rotor);
    });

    drone.scale.set(0.9, 0.9, 0.9);
    return drone;
}

// -----------------------------------------------------------------------------
// WORLD ENTITIES RENDERING (Neutral Gray Obstacles, Beacons, Trajectories)
// -----------------------------------------------------------------------------
function renderWorldEntities() {
    // 1. Clear existing obstacle & halo meshes
    obstacleMeshes.forEach(m => scene.remove(m));
    haloMeshes.forEach(h => scene.remove(h));
    obstacleMeshes = [];
    haloMeshes = [];

    // 2. Render Obstacles (Cylinders & Spheres)
    state.world.obstacles.forEach((obs) => {
        if (obs.enabled === false) return;

        const isSphere = obs.type === 'sphere';
        const obsMat = new THREE.MeshStandardMaterial({
            color: 0x94a3b8,
            roughness: 0.55,
            metalness: 0.15
        });
        const edgeMat = new THREE.LineBasicMaterial({ color: 0x334155, linewidth: 1 });

        if (isSphere) {
            const cz = obs.cz !== undefined ? obs.cz : 40.0;
            const geom = new THREE.SphereGeometry(obs.radius, 32, 24);
            const mesh = new THREE.Mesh(geom, obsMat);
            mesh.position.set(obs.cx, cz, obs.cy);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            scene.add(mesh);
            obstacleMeshes.push(mesh);

            // Subtle dark equator/meridian ring outline
            const ringGeom = new THREE.RingGeometry(obs.radius - 0.05, obs.radius + 0.05, 32);
            const ringMat = new THREE.MeshBasicMaterial({ color: 0x475569, side: THREE.DoubleSide });
            const ring = new THREE.Mesh(ringGeom, ringMat);
            ring.rotation.x = Math.PI / 2;
            ring.position.copy(mesh.position);
            scene.add(ring);
            obstacleMeshes.push(ring);

            // Safety Margin Halo (Outer buffer zone)
            const haloGeom = new THREE.SphereGeometry(obs.radius + state.world.margin, 20, 16);
            const haloMat = new THREE.MeshBasicMaterial({
                color: 0xf59e0b,
                wireframe: true,
                transparent: true,
                opacity: state.layers.showHalos ? 0.25 : 0.0
            });
            const halo = new THREE.Mesh(haloGeom, haloMat);
            halo.position.copy(mesh.position);
            scene.add(halo);
            haloMeshes.push(halo);
        } else {
            // Cylindrical Obstacle
            const geom = new THREE.CylinderGeometry(obs.radius, obs.radius, obs.height, 32);
            const mesh = new THREE.Mesh(geom, obsMat);
            mesh.position.set(obs.cx, obs.height / 2, obs.cy);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            scene.add(mesh);
            obstacleMeshes.push(mesh);

            // Dark Top & Bottom Edge Outlines
            const edges = new THREE.EdgesGeometry(geom, 25);
            const edgeLines = new THREE.LineSegments(edges, edgeMat);
            edgeLines.position.copy(mesh.position);
            scene.add(edgeLines);
            obstacleMeshes.push(edgeLines);

            // Safety Margin Halo
            const haloGeom = new THREE.CylinderGeometry(obs.radius + state.world.margin, obs.radius + state.world.margin, obs.height, 24);
            const haloMat = new THREE.MeshBasicMaterial({
                color: 0xf59e0b,
                wireframe: true,
                transparent: true,
                opacity: state.layers.showHalos ? 0.22 : 0.0
            });
            const halo = new THREE.Mesh(haloGeom, haloMat);
            halo.position.copy(mesh.position);
            scene.add(halo);
            haloMeshes.push(halo);
        }
    });

    // 3. Render Start (Red) & Goal (Green) Markers with Text Labels
    if (sourceBeacon) scene.remove(sourceBeacon);
    if (targetBeacon) scene.remove(targetBeacon);

    sourceBeacon = createWaypointMarker(state.world.source, 0xef4444, 'Start');
    targetBeacon = createWaypointMarker(state.world.target, 0x10b981, 'Goal');
    scene.add(sourceBeacon);
    scene.add(targetBeacon);

    // Place UAV at start location
    const startVec = simToThree(...state.world.source);
    droneMesh.position.copy(startVec);
}

function createWaypointMarker(pos, colorHex, labelText) {
    const group = new THREE.Group();
    const vec = simToThree(...pos);

    // Ground dot & thin stem
    const dotGeom = new THREE.CylinderGeometry(2.0, 2.0, 0.2, 16);
    const dotMat = new THREE.MeshBasicMaterial({ color: colorHex });
    const dot = new THREE.Mesh(dotGeom, dotMat);
    dot.position.set(vec.x, 0.1, vec.z);
    group.add(dot);

    const stemPts = [new THREE.Vector3(vec.x, 0, vec.z), new THREE.Vector3(vec.x, vec.y, vec.z)];
    const stemGeom = new THREE.BufferGeometry().setFromPoints(stemPts);
    const stemMat = new THREE.LineBasicMaterial({ color: colorHex, linewidth: 1 });
    group.add(new THREE.Line(stemGeom, stemMat));

    // Spherical marker ball
    const sphereGeom = new THREE.SphereGeometry(2.4, 20, 16);
    const sphereMat = new THREE.MeshStandardMaterial({
        color: colorHex,
        roughness: 0.3,
        metalness: 0.2
    });
    const sphere = new THREE.Mesh(sphereGeom, sphereMat);
    sphere.position.copy(vec);
    group.add(sphere);

    // Clean Text Label ("Start" or "Goal")
    const labelSprite = makeTextSprite(labelText, {
        fontSize: 32,
        fontColor: '#0f172a',
        fontWeight: '700',
        scale: [14, 7, 1]
    });
    labelSprite.position.set(vec.x, vec.y + 4.8, vec.z);
    group.add(labelSprite);

    return group;
}

// -----------------------------------------------------------------------------
// TRAJECTORY & EXPLORATION PATH VISUALIZATION
// -----------------------------------------------------------------------------
function clearAllTrajectories() {
    if (pathMesh) {
        scene.remove(pathMesh);
        pathMesh = null;
    }
    if (exploredPointCloud) {
        scene.remove(exploredPointCloud);
        exploredPointCloud = null;
    }
    Object.keys(multiPathMeshes).forEach(k => {
        if (multiPathMeshes[k]) {
            scene.remove(multiPathMeshes[k]);
            delete multiPathMeshes[k];
        }
    });
}

function renderTrajectory(pathCoords, exploredNodes = []) {
    clearAllTrajectories();

    if (!pathCoords || pathCoords.length < 2) return;

    // Build Continuous Smooth Flight Path Tube
    const points = pathCoords.map(p => simToThree(p[0], p[1], p[2]));
    const curve = new THREE.CatmullRomCurve3(points);
    const tubeGeom = new THREE.TubeGeometry(curve, Math.max(30, points.length * 4), 0.55, 8, false);

    const pathColor = ALGO_COLORS[state.activePlanner] || 0x0284c7;

    const tubeMat = new THREE.MeshStandardMaterial({
        color: pathColor,
        roughness: 0.3,
        metalness: 0.4,
        transparent: true,
        opacity: state.layers.showPath ? 0.95 : 0.0
    });
    pathMesh = new THREE.Mesh(tubeGeom, tubeMat);
    scene.add(pathMesh);

    // Explored Nodes (Graph Search Frontier) for A*
    const isAStar = (state.activePlanner === 'astar' || state.activePlanner === 'hybrid_astar_mappo');
    if (isAStar && exploredNodes && exploredNodes.length > 0 && state.layers.showExplored) {
        const pGeom = new THREE.BufferGeometry();
        const pCoords = [];
        exploredNodes.forEach(pt => {
            const v = simToThree(pt[0], pt[1], pt[2]);
            pCoords.push(v.x, v.y, v.z);
        });
        pGeom.setAttribute('position', new THREE.Float32BufferAttribute(pCoords, 3));
        const pMat = new THREE.PointsMaterial({
            color: 0x64748b,
            size: 1.6,
            transparent: true,
            opacity: 0.45
        });
        exploredPointCloud = new THREE.Points(pGeom, pMat);
        scene.add(exploredPointCloud);
    }
}

function renderMultiTrajectories(comparisonList) {
    clearAllTrajectories();

    comparisonList.forEach(item => {
        const key = item.key;
        const coords = item.path;
        if (!coords || coords.length < 2) return;

        const points = coords.map(p => simToThree(p[0], p[1], p[2]));
        const curve = new THREE.CatmullRomCurve3(points);
        const tubeGeom = new THREE.TubeGeometry(curve, Math.max(30, points.length * 4), 0.65, 8, false);
        const colorHex = ALGO_COLORS[key] || 0x64748B;

        const isVisible = (state.visiblePlanners[key] !== false) && state.layers.showPath;
        const tubeMat = new THREE.MeshStandardMaterial({
            color: colorHex,
            roughness: 0.3,
            metalness: 0.4,
            transparent: true,
            opacity: isVisible ? 0.95 : 0.0
        });
        const mesh = new THREE.Mesh(tubeGeom, tubeMat);
        mesh.visible = isVisible;
        scene.add(mesh);
        multiPathMeshes[key] = mesh;
    });
}

// -----------------------------------------------------------------------------
// BACKEND REST API CALLS
// -----------------------------------------------------------------------------
async function fetchWorldConfig() {
    try {
        const res = await fetch('/api/world');
        const data = await res.json();
        state.world.bounds = data.bounds;
        state.world.obstacles = data.obstacles.map(o => ({ ...o, enabled: true }));
        state.world.source = data.default_source;
        state.world.target = data.default_target;
        state.world.margin = data.safety_margin;

        document.getElementById('src-x').value = state.world.source[0];
        document.getElementById('src-y').value = state.world.source[1];
        document.getElementById('src-z').value = state.world.source[2];
        document.getElementById('tgt-x').value = state.world.target[0];
        document.getElementById('tgt-y').value = state.world.target[1];
        document.getElementById('tgt-z').value = state.world.target[2];

        setBackendStatus(true);
    } catch (e) {
        console.warn('Backend connection failed; falling back to local preset.', e);
        state.world.obstacles = JSON.parse(JSON.stringify(PRESETS.benchmark));
        setBackendStatus(false);
    }
}

function setBackendStatus(online) {
    const el = document.getElementById('backend-status');
    if (!el) return;
    if (online) {
        el.innerHTML = '<span class="status-dot online"></span><span class="badge-label">Backend:</span><span class="badge-value">Connected</span>';
    } else {
        el.innerHTML = '<span class="status-dot offline"></span><span class="badge-label">Backend:</span><span class="badge-value">Offline</span>';
    }
}

function formatStatusBadge(rawStatus) {
    if (!rawStatus) return { label: 'Standby', cls: 'tag-neutral' };
    const s = String(rawStatus).toUpperCase();
    if (s.includes('GOAL') || s.includes('SUCCESS') || s.includes('PATH_FOUND')) {
        return { label: 'Success', cls: 'tag-success' };
    }
    if (s.includes('STALL') || s.includes('BLOCK')) {
        return { label: 'Intervened', cls: 'tag-warning' };
    }
    if (s.includes('TIMEOUT') || s.includes('INCOMPLETE')) {
        return { label: 'Incomplete', cls: 'tag-warning' };
    }
    if (s.includes('NO_PATH')) {
        return { label: 'No Path', cls: 'tag-danger' };
    }
    return { label: rawStatus, cls: 'tag-neutral' };
}

async function runPlanning() {
    if (state.activePlanner === 'compare_all' || state.compareMode) {
        return runComparison();
    }

    const activeObstacles = state.world.obstacles.filter(o => o.enabled !== false);
    const payload = {
        planner: state.activePlanner,
        start: state.world.source,
        target: state.world.target,
        obstacles: activeObstacles,
        margin: state.world.margin
    };

    updatePlannerBadgeUI();
    const btnPlan = document.getElementById('btn-plan-run');
    btnPlan.disabled = true;
    btnPlan.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Planning...';

    try {
        const res = await fetch('/api/plan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const planData = await res.json();
        state.plannedTrajectory = planData;
        state.currentStepIdx = 0;

        const statusBadge = document.getElementById('badge-mission-status');
        if (statusBadge) {
            const st = formatStatusBadge(planData.status);
            statusBadge.innerText = `Status: ${st.label}`;
        }

        if (!planData.path || planData.path.length < 2) {
            renderTrajectory([], planData.explored_nodes || []);
            document.getElementById('step-counter').innerText = `Status: ${planData.status || 'No Path'}`;
            state.isPlaying = false;
            document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-play"></i>';
            return;
        }

        const stepsCount = planData.path.length - 1;
        document.getElementById('step-counter').innerText = `Simulation Step: 0 / ${stepsCount}`;
        const slider = document.getElementById('timeline-slider');
        slider.max = stepsCount;
        slider.value = 0;

        renderTrajectory(planData.path, planData.explored_nodes);
        updateDecisionInspector(0);

        state.isPlaying = true;
        document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-pause"></i>';
    } catch (e) {
        console.error('Error planning trajectory:', e);
    } finally {
        btnPlan.disabled = false;
        btnPlan.innerHTML = '<i class="fa-solid fa-play"></i> Plan &amp; Fly';
    }
}

async function runComparison() {
    state.compareMode = true;
    state.activePlanner = 'compare_all';
    const plannerSel = document.getElementById('planner-selector');
    if (plannerSel) plannerSel.value = 'compare_all';

    const btnCompare = document.getElementById('btn-compare-all');
    if (btnCompare) {
        btnCompare.classList.add('active');
        btnCompare.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Comparing...';
    }
    const btnPlan = document.getElementById('btn-plan-run');
    if (btnPlan) btnPlan.disabled = true;

    // Toggle inspector panels to show Comparison view
    const singleViews = document.querySelectorAll('#decision-inspector-panel .section-group');
    singleViews.forEach(el => {
        if (!el.closest('#inspector-compare-view')) el.style.display = 'none';
    });
    const compareView = document.getElementById('inspector-compare-view');
    if (compareView) compareView.style.display = 'block';

    try {
        const activeObstacles = state.world.obstacles.filter(o => o.enabled !== false);
        const res = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start: state.world.source,
                target: state.world.target,
                obstacles: activeObstacles,
                margin: state.world.margin
            })
        });
        const data = await res.json();
        state.comparisonResults = data.comparison || [];

        // Render all trajectories in 3D
        renderMultiTrajectories(state.comparisonResults);

        // Update Comparison Cards in Inspector
        const cardsContainer = document.getElementById('compare-cards-container');
        if (cardsContainer) {
            cardsContainer.innerHTML = '';
            state.comparisonResults.forEach(r => {
                const hex = ALGO_COLORS[r.key];
                const color = hex ? '#' + hex.toString(16).padStart(6, '0') : '#64748B';
                const st = formatStatusBadge(r.status);
                const isOurs = (r.key === 'hybrid_astar_mappo');
                const card = document.createElement('div');
                card.className = 'comp-metrics-card';
                card.innerHTML = `
                    <div class="comp-metrics-header" style="border-left: 3px solid ${color};">
                        <div class="comp-header-left">
                            <span class="comp-color-dot" style="background: ${color};"></span>
                            <span class="comp-algo-title">${r.algorithm}</span>
                            ${isOurs ? '<span class="badge-ours-pill">Ours</span>' : ''}
                        </div>
                        <span class="status-tag ${st.cls}"><span class="tag-bullet"></span>${st.label}</span>
                    </div>
                    <div class="comp-metrics-grid">
                        <div class="comp-metric-cell">
                            <span class="comp-metric-label">PATH LENGTH</span>
                            <span class="comp-metric-value">${r.path_length_m !== undefined ? r.path_length_m.toFixed(1) + ' m' : '--'}</span>
                        </div>
                        <div class="comp-metric-cell">
                            <span class="comp-metric-label">STEPS</span>
                            <span class="comp-metric-value">${r.execution_steps || 0}</span>
                        </div>
                        <div class="comp-metric-cell">
                            <span class="comp-metric-label">PLAN TIME</span>
                            <span class="comp-metric-value">${r.planning_time_ms !== undefined ? r.planning_time_ms.toFixed(2) + ' ms' : '--'}</span>
                        </div>
                        <div class="comp-metric-cell">
                            <span class="comp-metric-label">INTERVENTIONS</span>
                            <span class="comp-metric-value">${r.safety_interventions || 0}</span>
                        </div>
                    </div>
                `;
                cardsContainer.appendChild(card);
            });
        }

        // Set timeline to primary algorithm (hybrid or first with valid path)
        const primary = state.comparisonResults.find(r => r.key === 'hybrid_astar_mappo' && r.path && r.path.length > 1) || state.comparisonResults[0] || {};
        if (primary && primary.path && primary.path.length > 1) {
            state.plannedTrajectory = {
                status: primary.status,
                planner: primary.key,
                path: primary.path,
                steps_telemetry: primary.steps_telemetry || []
            };
            const stepsCount = primary.path.length - 1;
            document.getElementById('step-counter').innerText = `Simulation Step: 0 / ${stepsCount}`;
            const slider = document.getElementById('timeline-slider');
            slider.max = stepsCount;
            slider.value = 0;
            state.currentStepIdx = 0;
            state.isPlaying = true;
            document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-pause"></i>';
        }

        const badge = document.getElementById('badge-planner-name');
        if (badge) badge.innerText = 'Mode: Comparative Benchmark';
        const statusBadge = document.getElementById('badge-mission-status');
        if (statusBadge) {
            statusBadge.innerText = 'Status: Complete';
        }
    } catch (e) {
        console.error('Error running comparison:', e);
    } finally {
        if (btnCompare) {
            btnCompare.innerHTML = '<i class="fa-solid fa-layer-group"></i> Compare All';
        }
        if (btnPlan) btnPlan.disabled = false;
    }
}

function switchToSinglePlanner(plannerKey) {
    state.compareMode = false;
    state.activePlanner = plannerKey;
    const btnCompare = document.getElementById('btn-compare-all');
    if (btnCompare) btnCompare.classList.remove('active');

    // Restore single inspector view
    const singleViews = document.querySelectorAll('#decision-inspector-panel .section-group');
    singleViews.forEach(el => {
        if (!el.closest('#inspector-compare-view')) el.style.display = 'block';
    });
    const compareView = document.getElementById('inspector-compare-view');
    if (compareView) compareView.style.display = 'none';

    clearAllTrajectories();
    runPlanning();
}

function updatePlannerBadgeUI() {
    const names = {
        'hybrid_astar_mappo': 'Hierarchical A* + MAPPO',
        'astar': '3D A* Baseline',
        'rrt_star': '3D RRT* (Optimal Tree)',
        'sapf': '3D SAPF (Adaptive Field)',
        'mappo_residual': 'APF-Residual MAPPO',
        'mappo_pure': 'Pure MAPPO Policy',
        'ippo_residual': 'IPPO Residual',
        'ippo_pure': 'IPPO Pure',
        'maddpg_residual': 'MADDPG Residual',
        'maddpg_pure': 'MADDPG Pure',
        'cppo': 'Centralized PPO (CPPO)',
        'compare_all': 'Comparative Benchmark'
    };
    const name = names[state.activePlanner] || state.activePlanner;
    const badge = document.getElementById('badge-planner-name');
    if (badge) badge.innerText = `Planner: ${name}`;
    const infoName = document.getElementById('info-planner-name');
    if (infoName) infoName.innerText = name;
}

// -----------------------------------------------------------------------------
// DECISION INSPECTOR & LIVE TELEMETRY
// -----------------------------------------------------------------------------
function updateDecisionInspector(stepIdx) {
    if (!state.plannedTrajectory || !state.plannedTrajectory.steps_telemetry) return;
    const telem = state.plannedTrajectory.steps_telemetry[stepIdx];
    if (!telem) return;

    // 1. UAV Telemetry Grid
    document.getElementById('telem-pos').innerText = `(${telem.pos[0].toFixed(1)}, ${telem.pos[1].toFixed(1)}, ${telem.pos[2].toFixed(1)})`;
    document.getElementById('telem-speed').innerText = `${telem.speed.toFixed(1)}`;
    document.getElementById('telem-goal-dist').innerText = `${telem.dist_to_target.toFixed(1)}`;
    document.getElementById('telem-clearance').innerText = `${telem.min_clearance.toFixed(1)}`;

    // 2. Observation Vector
    const exp = telem.explanation || {};
    const tVec = exp.observation_target_rel || [
        (state.world.target[0] - telem.pos[0]) / 200.0,
        (state.world.target[1] - telem.pos[1]) / 200.0,
        (state.world.target[2] - telem.pos[2]) / 200.0
    ];
    document.getElementById('obs-target-rel').innerText = `[${formatFloat(tVec[0])}, ${formatFloat(tVec[1])}, ${formatFloat(tVec[2])}]`;

    const vRel = [telem.speed / 15.0, 0.0, 0.0];
    document.getElementById('obs-vel-rel').innerText = `[${formatFloat(vRel[0])}, ${formatFloat(vRel[1])}, ${formatFloat(vRel[2])}]`;

    const oVec = exp.observation_obstacle_rel || [0.42, 0.91, 0.0];
    document.getElementById('obs-obs-rel').innerText = `[${formatFloat(oVec[0])}, ${formatFloat(oVec[1])}, ${formatFloat(oVec[2] || 0.0)}]`;

    // 3. Selected Action
    const actVec = exp.neural_action || exp.executed_control || [0.0, 0.0, 0.0];
    document.getElementById('act-delta-v').innerText = `[${formatFloat(actVec[0])}, ${formatFloat(actVec[1])}, ${formatFloat(actVec[2])}]`;
    document.getElementById('act-speed-cmd').innerText = `${telem.speed.toFixed(1)}`;

    // 4. Planner Info
    const totalSteps = state.plannedTrajectory.path ? state.plannedTrajectory.path.length - 1 : 0;
    document.getElementById('info-step-count').innerText = `${stepIdx} / ${totalSteps}`;

    const safetyEl = document.getElementById('info-safety-status');
    if (telem.safety_intervened) {
        safetyEl.innerHTML = '<span class="status-badge-inline danger">INTERVENTION TRIGGERED</span>';
    } else {
        safetyEl.innerHTML = '<span class="status-badge-inline success">Not Active</span>';
    }

    // 5. Decision Explanation Subviews
    const isTreeSearch = (state.activePlanner === 'astar' || state.activePlanner === 'rrt_star');
    const astarView = document.getElementById('inspector-astar-view');
    const rlView = document.getElementById('inspector-rl-view');
    if (astarView) astarView.style.display = isTreeSearch ? 'block' : 'none';
    if (rlView) rlView.style.display = isTreeSearch ? 'none' : 'block';

    if (isTreeSearch) {
        if (exp.g_cost !== undefined) document.getElementById('astar-g').innerText = exp.g_cost.toFixed(1);
        if (exp.h_cost !== undefined) document.getElementById('astar-h').innerText = exp.h_cost.toFixed(1);
        if (exp.f_cost !== undefined) document.getElementById('astar-f').innerText = exp.f_cost.toFixed(1);
        if (exp.direction_chosen) {
            document.getElementById('astar-decision-note').innerText =
                `${exp.direction_chosen} | Explored: ${exp.explored_nodes_count || 0} tree/frontier nodes`;
        }
    } else {
        const hasNominal = Boolean(exp.apf_nominal);
        const apfRow = document.getElementById('row-apf-nom');
        if (apfRow) apfRow.style.display = hasNominal ? 'flex' : 'none';
        if (hasNominal && exp.apf_nominal) {
            document.getElementById('val-apf-nom').innerText =
                `[${formatFloat(exp.apf_nominal[0])}, ${formatFloat(exp.apf_nominal[1])}, ${formatFloat(exp.apf_nominal[2])}]`;
        }
        if (exp.neural_action) {
            document.getElementById('val-neural-out').innerText =
                `[${formatFloat(exp.neural_action[0])}, ${formatFloat(exp.neural_action[1])}, ${formatFloat(exp.neural_action[2])}]`;
        }
        if (exp.executed_control) {
            document.getElementById('val-effective-cmd').innerText =
                `[${formatFloat(exp.executed_control[0])}, ${formatFloat(exp.executed_control[1])}, ${formatFloat(exp.executed_control[2])}]`;
        }
    }
}

function formatFloat(v) {
    if (v === undefined || v === null || isNaN(v)) return '+0.00';
    return (v >= 0 ? '+' : '') + Number(v).toFixed(2);
}

// -----------------------------------------------------------------------------
// ANIMATION & FLIGHT PLAYBACK LOOP
// -----------------------------------------------------------------------------
let lastTime = 0;
function animateScene(time = 0) {
    requestAnimationFrame(animateScene);
    const delta = (time - lastTime) / 1000.0;
    lastTime = time;

    // Spin quadrotor blades
    rotors.forEach(r => {
        r.rotation.y += 0.45;
    });

    // Playback flight path
    if (state.isPlaying && state.plannedTrajectory && state.plannedTrajectory.path && state.plannedTrajectory.path.length > 1) {
        const path = state.plannedTrajectory.path;
        state.currentStepIdx += delta * 12.0 * state.playSpeed;

        if (state.currentStepIdx >= path.length - 1) {
            state.currentStepIdx = path.length - 1;
            state.isPlaying = false;
            document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-play"></i>';
        }

        const currIdx = Math.floor(state.currentStepIdx);
        const nextIdx = Math.min(currIdx + 1, path.length - 1);
        const alpha = state.currentStepIdx - currIdx;

        const p1 = simToThree(...path[currIdx]);
        const p2 = simToThree(...path[nextIdx]);
        droneMesh.position.lerpVectors(p1, p2, alpha);

        // Orient UAV forward in flight direction
        const dir = new THREE.Vector3().subVectors(p2, p1).normalize();
        if (dir.lengthSq() > 1e-4) {
            const yaw = Math.atan2(dir.x, dir.z);
            droneMesh.rotation.y = yaw;
        }

        document.getElementById('timeline-slider').value = currIdx;
        document.getElementById('step-counter').innerText = `Simulation Step: ${currIdx} / ${path.length - 1}`;
        updateDecisionInspector(currIdx);
    }

    controls.update();
    renderer.render(scene, camera);
}

// -----------------------------------------------------------------------------
// EVENT LISTENERS & UI INTERACTIONS
// -----------------------------------------------------------------------------
function setupEventListeners() {
    // Planner Selector Change
    const plannerSel = document.getElementById('planner-selector');
    plannerSel.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'compare_all') {
            runComparison();
        } else {
            switchToSinglePlanner(val);
        }
    });

    // Plan & Fly Button
    document.getElementById('btn-plan-run').addEventListener('click', runPlanning);

    // Compare All Button
    const btnCompareAll = document.getElementById('btn-compare-all');
    if (btnCompareAll) {
        btnCompareAll.addEventListener('click', () => {
            runComparison();
        });
    }

    // Individual Algorithm Checkboxes in Compare View (Instant Toggle)
    document.querySelectorAll('.comp-check-item input').forEach(chk => {
        chk.addEventListener('change', (e) => {
            const algoKey = e.target.getAttribute('data-algo');
            state.visiblePlanners[algoKey] = e.target.checked;
            if (multiPathMeshes[algoKey]) {
                multiPathMeshes[algoKey].visible = e.target.checked;
            }
        });
    });

    // Playback Controls
    document.getElementById('btn-play-pause').addEventListener('click', () => {
        state.isPlaying = !state.isPlaying;
        document.getElementById('btn-play-pause').innerHTML = state.isPlaying
            ? '<i class="fa-solid fa-pause"></i>'
            : '<i class="fa-solid fa-play"></i>';
    });

    document.getElementById('btn-stop-sim').addEventListener('click', () => {
        state.isPlaying = false;
        document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-play"></i>';
    });

    document.getElementById('btn-reset-sim').addEventListener('click', () => {
        state.isPlaying = false;
        state.currentStepIdx = 0;
        if (state.plannedTrajectory && state.plannedTrajectory.path && state.plannedTrajectory.path.length > 0) {
            const p = simToThree(...state.plannedTrajectory.path[0]);
            droneMesh.position.copy(p);
            updateDecisionInspector(0);
        }
        document.getElementById('timeline-slider').value = 0;
        const total = state.plannedTrajectory && state.plannedTrajectory.path ? state.plannedTrajectory.path.length - 1 : 0;
        document.getElementById('step-counter').innerText = `Simulation Step: 0 / ${total}`;
        document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-play"></i>';
    });

    // Timeline Slider Scrubbing
    document.getElementById('timeline-slider').addEventListener('input', (e) => {
        state.isPlaying = false;
        state.currentStepIdx = parseInt(e.target.value);
        if (state.plannedTrajectory && state.plannedTrajectory.path) {
            const p = simToThree(...state.plannedTrajectory.path[state.currentStepIdx]);
            droneMesh.position.copy(p);
            updateDecisionInspector(state.currentStepIdx);
            const total = state.plannedTrajectory.path.length - 1;
            document.getElementById('step-counter').innerText = `Simulation Step: ${state.currentStepIdx} / ${total}`;
        }
        document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-play"></i>';
    });

    // Scenario Preset Selection
    document.getElementById('preset-selector').addEventListener('change', (e) => {
        const key = e.target.value;
        if (PRESETS[key]) {
            state.world.obstacles = JSON.parse(JSON.stringify(PRESETS[key]));
            updateObstacleListUI();
            renderWorldEntities();
            runPlanning();
        }
    });

    // Add Obstacle Controls (Both Cylinder and Sphere)
    document.getElementById('btn-add-cylinder').addEventListener('click', () => {
        const nextId = state.world.obstacles.length;
        state.world.obstacles.push({
            id: nextId,
            type: 'cylinder',
            cx: Math.round(50 + Math.random() * 100),
            cy: Math.round(50 + Math.random() * 100),
            radius: 12.0,
            height: 80.0,
            enabled: true
        });
        updateObstacleListUI();
        renderWorldEntities();
        runPlanning();
    });

    document.getElementById('btn-add-sphere').addEventListener('click', () => {
        const nextId = state.world.obstacles.length;
        state.world.obstacles.push({
            id: nextId,
            type: 'sphere',
            cx: Math.round(50 + Math.random() * 100),
            cy: Math.round(50 + Math.random() * 100),
            cz: 40.0,
            radius: 16.0,
            enabled: true
        });
        updateObstacleListUI();
        renderWorldEntities();
        runPlanning();
    });

    // Mission Endpoints Apply Button
    document.getElementById('btn-update-endpoints').addEventListener('click', () => {
        state.world.source = [
            parseFloat(document.getElementById('src-x').value) || 20,
            parseFloat(document.getElementById('src-y').value) || 20,
            parseFloat(document.getElementById('src-z').value) || 10
        ];
        state.world.target = [
            parseFloat(document.getElementById('tgt-x').value) || 180,
            parseFloat(document.getElementById('tgt-y').value) || 180,
            parseFloat(document.getElementById('tgt-z').value) || 10
        ];
        renderWorldEntities();
        runPlanning();
    });

    // Mission Endpoints Randomize Button
    const btnRandom = document.getElementById('btn-random-endpoints');
    if (btnRandom) {
        btnRandom.addEventListener('click', async () => {
            try {
                btnRandom.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Randomizing...';
                const activeObstacles = state.world.obstacles.filter(o => o.enabled !== false);
                const res = await fetch('/api/random_endpoints', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        obstacles: activeObstacles,
                        margin: state.world.margin
                    })
                });
                const data = await res.json();
                if (data.source && data.target) {
                    state.world.source = data.source;
                    state.world.target = data.target;
                    document.getElementById('src-x').value = data.source[0];
                    document.getElementById('src-y').value = data.source[1];
                    document.getElementById('src-z').value = data.source[2];
                    document.getElementById('tgt-x').value = data.target[0];
                    document.getElementById('tgt-y').value = data.target[1];
                    document.getElementById('tgt-z').value = data.target[2];
                    renderWorldEntities();
                    runPlanning();
                }
            } catch (err) {
                console.error('Error randomizing endpoints:', err);
            } finally {
                btnRandom.innerHTML = '<i class="fa-solid fa-shuffle"></i> Randomize';
            }
        });
    }

    // Visualization Layer Toggles
    document.getElementById('chk-show-path').addEventListener('change', (e) => {
        state.layers.showPath = e.target.checked;
        if (pathMesh) pathMesh.visible = e.target.checked;
    });

    document.getElementById('chk-show-explored').addEventListener('change', (e) => {
        state.layers.showExplored = e.target.checked;
        if (exploredPointCloud) exploredPointCloud.visible = e.target.checked;
    });

    document.getElementById('chk-show-halos').addEventListener('change', (e) => {
        state.layers.showHalos = e.target.checked;
        haloMeshes.forEach(h => {
            h.material.opacity = e.target.checked ? 0.22 : 0.0;
        });
    });

    // Camera Views (Top, Side, 3D)
    const btnTop = document.getElementById('btn-camera-top');
    const btnSide = document.getElementById('btn-camera-side');
    const btn3D = document.getElementById('btn-camera-reset');

    function setActiveViewBtn(btn) {
        [btnTop, btnSide, btn3D].forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }

    btnTop.addEventListener('click', () => {
        setActiveViewBtn(btnTop);
        camera.position.set(100, 320, 100);
        controls.target.set(100, 0, 100);
    });

    btnSide.addEventListener('click', () => {
        setActiveViewBtn(btnSide);
        camera.position.set(100, 35, 340);
        controls.target.set(100, 35, 100);
    });

    btn3D.addEventListener('click', () => {
        setActiveViewBtn(btn3D);
        camera.position.set(-90, 200, -90);
        controls.target.set(100, 35, 100);
    });

    // Benchmark Modal
    document.getElementById('btn-open-compare').addEventListener('click', openBenchmarkModal);
    document.getElementById('btn-close-modal').addEventListener('click', () => {
        document.getElementById('compare-modal').style.display = 'none';
    });
    document.getElementById('btn-run-all-benchmarks').addEventListener('click', runBenchmarkComparison);
    document.getElementById('btn-export-json').addEventListener('click', exportBenchmarkJSON);
}

// -----------------------------------------------------------------------------
// OBSTACLE LIST UI GENERATOR (Compact CAD-Style List)
// -----------------------------------------------------------------------------
function updateObstacleListUI() {
    const container = document.getElementById('obstacle-list-container');
    container.innerHTML = '';

    state.world.obstacles.forEach((obs, idx) => {
        const item = document.createElement('div');
        item.className = 'obstacle-item';

        const isSphere = obs.type === 'sphere';
        const typeLabel = isSphere ? `Sphere #${idx + 1}` : `Cylinder #${idx + 1}`;
        const subLabel = isSphere
            ? `C: (${obs.cx.toFixed(1)}, ${obs.cy.toFixed(1)}, ${(obs.cz || 40).toFixed(1)}) R: ${obs.radius}m`
            : `C: (${obs.cx.toFixed(1)}, ${obs.cy.toFixed(1)}) R: ${obs.radius}m H: ${obs.height}m`;

        item.innerHTML = `
            <div class="obs-item-content">
                <input type="checkbox" class="obs-enable-chk" data-idx="${idx}" ${obs.enabled !== false ? 'checked' : ''} title="Toggle Obstacle">
                <div class="obs-info">
                    <span class="obs-title">${typeLabel}</span>
                    <span class="obs-sub">${subLabel}</span>
                </div>
            </div>
            <button class="obs-del-btn" data-idx="${idx}" title="Delete Obstacle">
                <i class="fa-solid fa-trash"></i>
            </button>
        `;
        container.appendChild(item);
    });

    // Enable/Disable checkbox handler
    document.querySelectorAll('.obs-enable-chk').forEach(chk => {
        chk.addEventListener('change', (e) => {
            const idx = parseInt(e.target.getAttribute('data-idx'));
            state.world.obstacles[idx].enabled = e.target.checked;
            renderWorldEntities();
            runPlanning();
        });
    });

    // Delete obstacle handler
    document.querySelectorAll('.obs-del-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const idx = parseInt(e.currentTarget.getAttribute('data-idx'));
            state.world.obstacles.splice(idx, 1);
            updateObstacleListUI();
            renderWorldEntities();
            runPlanning();
        });
    });
}

// -----------------------------------------------------------------------------
// BENCHMARK MODAL & RESEARCH COMPARISON
// -----------------------------------------------------------------------------
let lastBenchmarkData = null;

async function openBenchmarkModal() {
    document.getElementById('compare-modal').style.display = 'flex';
    runBenchmarkComparison();
}

async function runBenchmarkComparison() {
    const tbody = document.getElementById('modal-table-body');
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:16px;">Executing matched planners on active 3D layout...</td></tr>';

    try {
        const activeObstacles = state.world.obstacles.filter(o => o.enabled !== false);
        const res = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start: state.world.source,
                target: state.world.target,
                obstacles: activeObstacles,
                margin: state.world.margin
            })
        });
        const data = await res.json();
        lastBenchmarkData = data.comparison;
        tbody.innerHTML = '';

        data.comparison.forEach(r => {
            const tr = document.createElement('tr');
            const st = formatStatusBadge(r.status);
            tr.innerHTML = `
                <td style="font-weight:600; color:#0f172a;">${r.algorithm}</td>
                <td><span class="status-tag ${st.cls}"><span class="tag-bullet"></span>${st.label}</span></td>
                <td class="mono">${r.path_length_m !== undefined ? r.path_length_m.toFixed(1) : '--'}</td>
                <td class="mono">${r.execution_steps || 0}</td>
                <td class="mono">${r.planning_time_ms !== undefined ? r.planning_time_ms.toFixed(2) : '--'}</td>
                <td class="mono">${r.min_clearance_m !== undefined ? r.min_clearance_m.toFixed(1) : '--'}</td>
                <td class="mono">${r.safety_interventions || 0}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        tbody.innerHTML = '<tr><td colspan="7" style="color:#ef4444;text-align:center;padding:16px;">Failed to fetch benchmark comparison.</td></tr>';
    }
}

function exportBenchmarkJSON() {
    if (!lastBenchmarkData) return;
    const blob = new Blob([JSON.stringify(lastBenchmarkData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `uav_benchmark_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}
