// app/predict/page.js
"use client";

import { useState } from "react";
import { useSearchParams } from "next/navigation";
import styles from "./predict.module.css";

export default function Predict() {
    const searchParams = useSearchParams();
    const model = searchParams.get("model"); // "esc10" ou "urbansound"
    const [file, setFile] = useState(null);
    const [url, setUrl] = useState("");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const modelName = model === "esc10" ? "ESC-10" : "UrbanSound8K";
    const apiPrefix = model === "esc10" ? "esc10" : "urbansound"; // pour construire l'URL de l'API

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleUrlChange = (e) => {
        setUrl(e.target.value);
    };

    const handleFileUpload = async () => {
        if (!file) return alert("Veuillez sélectionner un fichier.");
        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);
        try {
            const res = await fetch(`http://127.0.0.1:8000/predict/${apiPrefix}`, {
                method: "POST",
                body: formData,
            });
            const data = await res.json();
            setResult(data);
        } catch (error) {
            console.error("Erreur:", error);
            alert("Erreur de prédiction.");
        } finally {
            setLoading(false);
        }
    };

    const handleYoutubePredict = async () => {
        if (!url) return alert("Veuillez entrer un lien YouTube.");
        setLoading(true);
        try {
            const res = await fetch(
                `http://127.0.0.1:8000/predict_youtube/${apiPrefix}?url=${encodeURIComponent(url)}`,
                {
                    method: "POST",
                }
            );
            const data = await res.json();
            setResult(data);
        } catch (error) {
            console.error("Erreur:", error);
            alert("Erreur de prédiction.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={styles.container}>
            <header className={styles.header}>
                <h1>{modelName} Prédiction</h1>
                <p>Testez le modèle et découvrez l'univers sonore sélectionné</p>
            </header>
            <main className={styles.main}>
                <section className={styles.formSection}>
                    <h2>Uploader un fichier audio</h2>
                    <input type="file" onChange={handleFileChange} className={styles.input} />
                    <button onClick={handleFileUpload} className={styles.button} disabled={loading}>
                        {loading ? "Prédiction en cours..." : "Prédire"}
                    </button>
                </section>
                <section className={styles.formSection}>
                    <h2>Ou prédire via un lien YouTube</h2>
                    <input
                        type="text"
                        placeholder="Entrez l'URL YouTube"
                        value={url}
                        onChange={handleUrlChange}
                        className={styles.input}
                    />
                    <button onClick={handleYoutubePredict} className={styles.button} disabled={loading}>
                        {loading ? "Prédiction en cours..." : "Prédire"}
                    </button>
                </section>
                {result && result.all_confidences && (
                    <section className={styles.resultSection}>
                        <h2>Résultat</h2>
                        <div className={styles.card}>
                            <img
                                src={`/images/${apiPrefix}_${result.label}.png`}
                                alt={result.label}
                                className={styles.resultImage}
                            />
                            <div className={styles.resultText}>
                                <h3 className={styles.resultLabel}>{result.label}</h3>
                                <p className={styles.resultConfidence}>{result.confidence}</p>
                            </div>
                        </div>
                        <div className={styles.allPercentages}>
                            {Object.entries(result.all_confidences).map(([label, percentage]) => (
                                <div key={label} className={styles.percentageItem}>
                                    <span className={styles.percentageLabel}>{label}</span>
                                    <span className={styles.percentageValue}>{percentage}</span>
                                </div>
                            ))}
                        </div>
                    </section>
                )}
            </main>
            <footer className={styles.footer}>
                <p>Fait avec ❤️ par Amy, Mélissa, Farah, Rayan, Bahia et Dounia</p>
            </footer>
        </div>
    );
}
