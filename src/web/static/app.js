console.log("app.js loaded");

window.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predict-form");

    if (!form) {
        console.error("Form not found");
        return;
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        console.log("Predict clicked");

        const payload = {
            date: document.getElementById("date").value,
            temperature_c: Number(document.getElementById("temperature_c").value),
            humidity_percent: Number(document.getElementById("humidity_percent").value),
            wind_speed_ms: Number(document.getElementById("wind_speed_ms").value),
            sea_level_pressure_hpa: Number(
                document.getElementById("sea_level_pressure_hpa").value
            )
        };

        console.log("Payload:", payload);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                console.error("Prediction error:", data);
                document.getElementById("result").innerText =
                    "Prediction failed. Check console.";
                return;
            }

            document.getElementById("result").innerText =
                `Predicted Rainfall: ${data.prediction_mm.toFixed(2)} mm
Uncertainty: ±${data.uncertainty_mm.toFixed(2)} mm
Flood Risk Level: ${data.risk_level}`;

        } catch (err) {
            console.error("Request failed:", err);
            document.getElementById("result").innerText =
                "Server error. Check console.";
        }
    });
});
