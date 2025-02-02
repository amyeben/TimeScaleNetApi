// pages/index.js
"use client";

import { useRouter } from "next/navigation";
import styles from "./page.module.css";

export default function Home() {
  const router = useRouter();

  const handleSelect = (model) => {
    // Redirige vers la page de prédiction en passant le modèle dans l'URL
    router.push(`/predict?model=${model}`);
  };

  return (
      <div className={styles.container}>
        <header className={styles.header}>
          <h1>TimeScaleNet Prédiction</h1>
          <p>Choisissez votre univers sonore</p>
        </header>
        <main className={styles.main}>
          <button className={styles.bigButton} onClick={() => handleSelect("esc10")}>
            ESC-10
          </button>
          <button className={styles.bigButton} onClick={() => handleSelect("urbansound")}>
            UrbanSound8K
          </button>
        </main>
        <footer className={styles.footer}>
          <p>Fait avec ❤️ par Amy, Mélissa, Farah, Rayan, Bahia et Dounia</p>
        </footer>
      </div>
  );
}
