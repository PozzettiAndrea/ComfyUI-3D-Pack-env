import { app } from "/scripts/app.js"

// Locate our own /extensions/<pack>/ mount instead of naming the pack. The
// folder is the pack directory (or [project] name), so a fork or a rename
// changes it -- hardcoding "ComfyUI-3D-Pack" is what 404'd every viewer in
// this fork, which ships as ComfyUI-3D-Pack-enved.
const EXTENSION_FOLDER = (() => {
    const match = import.meta.url.match(/\/extensions\/([^/]+)\//)
    return match ? match[1] : "ComfyUI-3D-Pack-enved"
})()

class Visualizer {
    constructor(node, container, visualSrc) {
        this.node = node

        this.iframe = document.createElement('iframe')
        Object.assign(this.iframe, {
            scrolling: "no",
            overflow: "hidden",
        })
        // Size it here, not from a draw() callback. An iframe defaults to about
        // 300x150 regardless of its parent, so without this the viewer sits in a
        // small box in the corner of the node. The old canvas-widget code set
        // these every frame from draw(); that went away with the move to
        // addDOMWidget, and the styling had to come with it.
        Object.assign(this.iframe.style, {
            width: "100%",
            height: "100%",
            border: "0 none",
            display: "block",   // kill the inline-element baseline gap
        })
        this.iframe.src = `/extensions/${EXTENSION_FOLDER}/html/${visualSrc}.html`
        container.appendChild(this.iframe)
    }

    updateVisual(filepath) {
        const iframeDocument = this.iframe.contentWindow.document
        const previewScript = iframeDocument.getElementById('visualizer')
        previewScript.setAttribute("filepath", filepath)

        const timestamp = Date.now().toString()
        previewScript.setAttribute("timestamp", timestamp)
    }

    remove() {
        this.container.remove()
    }
}

function createVisualizer(node, inputName, typeName, inputData, app) {
    node.name = inputName

    // A DOM widget, not a canvas widget positioned by hand.
    //
    // This used to build an absolutely-positioned div, append it to
    // document.body, and re-compute left/top/width/height from canvas
    // coordinates on every draw. That injects into the shared page -- the
    // element outlives the node, sits in another pack's DOM, and has to
    // re-derive a transform ComfyUI already knows. addDOMWidget is the
    // sanctioned equivalent: ComfyUI owns placement, visibility, zoom and
    // teardown, and the element stays inside the node. Same approach as
    // ComfyUI-GeometryPack (js/mesh_preview_three.js:47).
    const container = document.createElement('div')
    container.id = `Comfy3D_${inputName}`
    Object.assign(container.style, {
        width: '100%',
        height: '100%',
        overflow: 'hidden',
    })

    node.visualizer = new Visualizer(node, container, typeName)

    const widget = node.addDOMWidget(inputName, typeName, container, {
        getValue() { return "" },
        setValue(_v) {},
    })
    widget.visualizer = container
    widget.parent = node

    // Height comes from computedHeight, NOT computeSize.
    //
    // ComfyUI lays a DOM widget out from widget.computedHeight; computeSize is
    // consulted for ordinary canvas widgets and ignored here. Left alone it
    // arrives as NaN, and the viewer then renders at a height unrelated to the
    // node -- width tracked the node correctly while height sat at a constant.
    // Measured in a browser: node 600x500 and 900x800 both produced a 1000px
    // tall frame until this was set.
    const WIDGET_CHROME = 68   // title bar + the node's own padding

    const fitToNode = () => {
        widget.computedHeight = Math.max(node.size[1] - WIDGET_CHROME, 160)
    }

    // computeSize stays consistent with it so anything that does read it (the
    // node's initial auto-size, for one) agrees with what is drawn.
    widget.computeSize = function (width) {
        return [width, Math.max(node.size[1] - WIDGET_CHROME, 160)]
    }

    node.updateParameters = (params) => {
        node.visualizer.updateVisual(params.filepath)
    }

    // Re-fit on every resize. Without this the element keeps whatever height it
    // was given when the node was created, so dragging the node's corner grew
    // the node and left the viewer behind.
    node.onResize = function (size) {
        const w = Math.max(size ? size[0] : this.size[0], 320)
        const h = Math.max(size ? size[1] : this.size[1], 240)
        this.size[0] = w
        this.size[1] = h
        fitToNode()
    }

    fitToNode()

    // Events for remove nodes
    node.onRemoved = () => {
        for (let w in node.widgets) {
            if (node.widgets[w].visualizer) {
                node.widgets[w].visualizer.remove()
            }
        }
    }


    return {
        widget: widget,
    }
}

function registerVisualizer(nodeType, nodeData, nodeClassName, typeName){
    if (nodeData.name == nodeClassName) {
        console.log("[3D Visualizer] Registering node: " + nodeData.name)

        const onNodeCreated = nodeType.prototype.onNodeCreated

        nodeType.prototype.onNodeCreated = async function() {
            const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined

            let Preview3DNode = app.graph._nodes.filter(
                (wi) => wi.type == nodeClassName
            )
            let nodeName = `Preview3DNode_${nodeClassName}`

            console.log(`[Comfy3D] Create: ${nodeName}`)

            const result = await createVisualizer.apply(this, [this, nodeName, typeName, {}, app])

            this.setSize([600, 500])

            return r
        }

        nodeType.prototype.onExecuted = async function(message) {
            if (message?.previews) {
                this.updateParameters(message.previews[0])
            }
        }
    }
}

app.registerExtension({
    // Namespaced under the pack's ComfyUI DisplayName. Extension names are
    // global: two packs registering the same name means one is silently
    // dropped, and "Mr.ForExample.Visualizer.GS" is inherited from upstream,
    // so the unmodified 3D-Pack and this fork would collide in one install.
    name: "3dpackenved.Visualizer",

    async init (app) {

    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        registerVisualizer(nodeType, nodeData, "[Comfy3D] Preview 3DGS", "gsVisualizer")
        registerVisualizer(nodeType, nodeData, "[Comfy3D] Preview 3DMesh", "threeVisualizer")
    },
})