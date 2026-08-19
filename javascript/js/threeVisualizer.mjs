import * as THREE from 'three';
//import { api } from '/scripts/ui/api.ts';
import {getRGBValue} from './sharedFunctions.mjs';

import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

const visualizer = document.getElementById("visualizer");
const container = document.getElementById( 'container' );
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");
const colorPicker = document.getElementById("color-picker");
const downloadButton = document.getElementById("download-button");

const renderer = new THREE.WebGLRenderer( { antialias: true } );
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( window.innerWidth, window.innerHeight );
container.appendChild( renderer.domElement );

const pmremGenerator = new THREE.PMREMGenerator( renderer );

// scene
const scene = new THREE.Scene();
scene.background = new THREE.Color( 0x000000 );
scene.environment = pmremGenerator.fromScene( new RoomEnvironment( renderer ), 0.04 ).texture;

const ambientLight = new THREE.AmbientLight( 0xffffff , 3.0 );

const camera = new THREE.PerspectiveCamera( 40, window.innerWidth / window.innerHeight, 1, 100 );
camera.position.set( 5, 2, 8 );
const pointLight = new THREE.PointLight( 0xffffff, 15 );
camera.add( pointLight );

const controls = new OrbitControls( camera, renderer.domElement );
controls.target.set( 0, 0.5, 0 );
controls.update();
controls.enablePan = true;
controls.enableDamping = true;

// Render on demand, not every frame.
//
// This loop used to call renderer.render() on every animation frame for the
// life of the node, whether or not anything had changed. Each Preview 3D node
// is its own iframe with its own WebGL context, so N nodes meant N scenes
// redrawing forever and the ComfyUI canvas fighting all of them for the main
// thread. Measured headless: one viewer node took the page from 61 fps to 7,
// four took it to 2.
//
// `dirty` is set by anything that changes the picture; controls.update()
// separately reports camera movement (and keeps reporting it while damping
// settles), and an animation mixer always needs frames.
let dirty = true;
let lastColorStyle = null;   // see the colour check in frameUpdate()
// No controls 'change' listener: OrbitControls dispatches it from update(),
// which under damping fires every frame forever (see cameraMoved below), so
// it would re-dirty the scene on a camera at rest. cameraMoved() covers real
// camera changes; `dirty` is for everything else (load, resize, colour).

// Decide "did the camera actually move?" ourselves rather than trusting
// controls.update()'s return value. With enableDamping it keeps reporting
// movement forever: its target test is a strict `distanceToSquared > 0`, and
// the damped offsets decay towards zero without ever reaching it, so the
// return value stays true and the 'change' event keeps firing on a camera
// that is visually at rest. Comparing against an epsilon settles.
const _lastCam = {
    pos: camera.position.clone(),
    quat: camera.quaternion.clone(),
    target: controls.target.clone(),
    zoom: camera.zoom,
};
const _CAM_EPS = 1e-10;

function cameraMoved() {
    const moved =
        camera.position.distanceToSquared( _lastCam.pos ) > _CAM_EPS ||
        ( 1 - Math.abs( camera.quaternion.dot( _lastCam.quat ) ) ) > _CAM_EPS ||
        controls.target.distanceToSquared( _lastCam.target ) > _CAM_EPS ||
        Math.abs( camera.zoom - _lastCam.zoom ) > 1e-6;
    if ( moved ) {
        _lastCam.pos.copy( camera.position );
        _lastCam.quat.copy( camera.quaternion );
        _lastCam.target.copy( controls.target );
        _lastCam.zoom = camera.zoom;
    }
    return moved;
}

// Handle window reseize event
window.onresize = function () {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );
    dirty = true;

};

const clock = new THREE.Clock();

var lastTimestamp = "";
var needUpdate = false;
let mixer;
let currentURL;
var url = location.protocol + '//' + location.host;

downloadButton.addEventListener('click', e => {
    window.open(currentURL, '_blank');
});

function frameUpdate() {

    var filepath = visualizer.getAttribute("filepath");
    var timestamp = visualizer.getAttribute("timestamp");
    if (timestamp == lastTimestamp){
        if (needUpdate){
            controls.update();          // must run every frame for damping
            const moved = cameraMoved();
            if (mixer !== undefined) {
                const delta = clock.getDelta();
                mixer.update(delta);
                dirty = true;          // an animation changes every frame
            }
            window.__probe = window.__probe || {frames:0, renders:0, dirty:0, moved:0};
            window.__probe.frames++;
            if (dirty) window.__probe.dirty++;
            if (moved) window.__probe.moved++;
            if (dirty || moved) {
                window.__probe.renders++;
                renderer.render( scene, camera );
                dirty = false;
            }
        }
        requestAnimationFrame( frameUpdate );
    } else {
        needUpdate = false;
        scene.clear();
        progressDialog.open = true;
        lastTimestamp = timestamp;
        main(filepath);
    }

    // Compare the picker's own string, not its parsed channels against
    // scene.background.
    //
    // getRGBValue returns sRGB 0..1 (rgb(128,128,128) -> 0.502), while
    // three.js colour-manages setStyle() and stores the LINEAR value (~0.216).
    // Those never compare equal, so this branch ran on every single frame --
    // re-applying the background and marking the scene dirty forever. That is
    // what kept each viewer redrawing at full rate with nothing happening:
    // measured one WebGL draw call per animation frame, per node, while idle.
    if (colorPicker.value !== lastColorStyle) {
        lastColorStyle = colorPicker.value;
        scene.background.setStyle(colorPicker.value);
        dirty = true;   // repainted by the gated render above
    }
}

const onProgress = function ( xhr ) {
    if ( xhr.lengthComputable ) {
        progressIndicator.value = xhr.loaded / xhr.total * 100;
    }
};
const onError = function ( e ) {
    console.error( e );
};

async function main(filepath="") {
    // Check if file name is valid
    if (/^.+\.[a-zA-Z]+$/.test(filepath)){

        // ComfyUI's own /view when the file is one it already serves
        // (output/input/temp), the pack's /viewfile only for paths outside
        // those. /view needs no client-IP allow-list, which is what made
        // /viewfile 404 for every browser that is not on the ComfyUI host --
        // through a tunnel, or simply from another machine on the LAN.
        const served = filepath.match(/[\\/](output|input|temp)[\\/](.*)$/);
        if (served) {
            const rel = served[2].replace(/\\/g, "/");
            const cut = rel.lastIndexOf("/");
            currentURL = url + '/view?' + new URLSearchParams({
                filename:  cut === -1 ? rel : rel.slice(cut + 1),
                subfolder: cut === -1 ? ""  : rel.slice(0, cut),
                type:      served[1],
            });
        } else {
            currentURL = url + '/viewfile?' + new URLSearchParams({"filepath": filepath});
        }

        var filepathSplit = filepath.split('.');
        var fileExt = filepathSplit.pop().toLowerCase();
        var filepathNoExt = filepathSplit.join(".");

        if (fileExt == "obj"){
            const loader = new OBJLoader();

            var mtlFolderpath = filepath.substring(0, Math.max(filepath.lastIndexOf("/"), filepath.lastIndexOf("\\"))) + "/";
            var mtlFilepath = filepathNoExt.replace(/^.*[\\\/]/, '') + ".mtl";

            const mtlLoader = new MTLLoader();
            mtlLoader.setPath(url + '/viewfile?' + new URLSearchParams({"filepath": mtlFolderpath}));
            mtlLoader.load( mtlFilepath, function ( mtl ) {
                mtl.preload();
                loader.setMaterials( mtl );
            }, onProgress, onError );

            loader.load( currentURL, function ( obj ) {
                obj.scale.setScalar( 5 );
                scene.add( obj );
                obj.traverse(node => {
                    if (node.material && node.material.map == null) {
                        node.material.vertexColors = true;
                    }
                  });

            }, onProgress, onError );

        } else if (fileExt == "glb") {
            const dracoLoader = new DRACOLoader();
            // No setDecoderPath: DRACOLoader defaults to '../libs/draco/gltf/'
            // resolved against its OWN module URL (DRACOLoader.js:22-23), which
            // now points at the vendored copy under lib/three/addons/libs/.
            // The old value pinned an unpkg URL at three@latest, so a Draco mesh
            // needed the network and could break on any upstream release.
            const loader = new GLTFLoader();
            loader.setDRACOLoader( dracoLoader );

            loader.load( currentURL, function ( gltf ) {
                const model = gltf.scene;
                //model.position.set( 1, 1, 0 );
                model.scale.set( 3, 3, 3 );

                scene.add( model );
                mixer = new THREE.AnimationMixer(model);
                gltf.animations.forEach((clip) => {
                    mixer.clipAction(clip).play();
                });

            }, onProgress, onError );

        } else if (fileExt == "ply") {

        } else {
            throw new Error(`File extension name has to be either .ply or .splat, got .${fileExt}`);
        }

        needUpdate = true;
        dirty = true;
    }

    scene.add( ambientLight );
    scene.add( camera );

    progressDialog.close();

    frameUpdate();
}

//main("C:/Users/reall/Softwares/ComfyUI_windows_portable/ComfyUI/output/MeshTest/Mesh_01.obj");
main();
