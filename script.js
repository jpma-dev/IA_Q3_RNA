let model;

// Cargar CSV
async function loadData() {
    const headers = [
        "Model", "Year", "Category", "Power(HP)", "Top speed(km/h)",
        "Torque(Nm)", "Displacement(ccm)", "Diameter(mm)", "Weight", "Price(USD)"
    ];

    const raw = await tf.data.csv("bikes.csv", {
        hasHeader: false,
        columnNames: headers
    }).toArray();

    const clean = raw.filter(d => {
        // Verificar que las columnas EXISTAN
        if (
            d["Power(HP)"] === undefined ||
            d["Displacement(ccm)"] === undefined ||
            d["Price(USD)"] === undefined
        ) {
            return false;
        }

        // Convertir
        const hp = Number(d["Power(HP)"]);
        const disp = Number(d["Displacement(ccm)"]);
        const price = Number(d["Price(USD)"]);

        // Validar valores numéricos reales
        return (
            !isNaN(hp) && hp > 0 &&
            !isNaN(disp) && disp > 0 &&
            !isNaN(price) && price > 0
        );
    })
    .map(d => ({
        hp: Number(d["Power(HP)"]),
        disp: Number(d["Displacement(ccm)"]),
        label: Number(d["Price(USD)"]) > 6000 ? 1 : 0
    }));

    console.log("Registros originales:", raw.length);
    console.log("Registros limpios:", clean.length);

    return clean;
}





// Crear modelo
function createModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [2], units: 8, activation: "relu" }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"]
    });

    return model;
}

async function trainModel() {
    const status = document.getElementById("status");
    const loader = document.getElementById("loader");
    const controls = document.getElementById("controls");

    status.innerText = "Cargando datos...";
    loader.style.display = "block"; // iniciar loader

    const dataset = await loadData();

    const xs = dataset.map(d => [d.hp, d.disp]);
    const ys = dataset.map(d => d.label);

    const xsTensor = tf.tensor2d(xs);
    const ysTensor = tf.tensor1d(ys);

    createModel();

    status.innerText = "Entrenando modelo...";

    await model.fit(xsTensor, ysTensor, {
        epochs: 20,
        batchSize: 16,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                const acc = logs.acc ?? logs.accuracy ?? 0;
                status.innerText = `Epoch ${epoch + 1} - Loss: ${logs.loss.toFixed(4)} - Accuracy: ${acc.toFixed(4)}`;
            }
        }
    });

    // Finaliza entrenamiento
    status.innerText = "Entrenamiento completado ✓";
    loader.style.display = "none"; // ocultar loader
    controls.classList.remove("hidden"); // mostrar formulario
}



async function predict() {
    const hp = Number(document.getElementById("hp").value);
    const disp = Number(document.getElementById("displacement").value);

    if (isNaN(hp) || isNaN(disp)) {
        alert("Ingrese valores válidos.");
        return;
    }

    const resultSpan = document.getElementById("result");

    // Usar tf.tidy para evitar que TensorFlow.js "congele" el primer resultado
    const prob = await tf.tidy(() => {
        const input = tf.tensor2d([[hp, disp]]);
        const output = model.predict(input);
        return output.data();
    });

    const prediction = prob[0] > 0.5 ? 
        "La motocicleta tendrá precio Caro" : 
        "La motocicleta tendrá precio Barato";

    resultSpan.innerText = prediction;
}

window.onload = () => {
    trainModel();
};