const API_BASE_URL = "https://movie-recommender-backend-a9wo.onrender.com";
let currentUser = "";

// ------------------- DOM ELEMENTS -------------------
const loginSection = document.getElementById("login-section");
const recommendSection = document.getElementById("recommend-section");
const usernameInput = document.getElementById("username");
const loginButton = document.getElementById("loginButton");
const logoutButton = document.getElementById("logoutButton");
const recommendationsDiv = document.getElementById("recommendations");
const historyDiv = document.getElementById("history");
const spinner = document.getElementById("spinner");

// ------------------- INITIALIZE FILTERS -------------------
window.addEventListener("DOMContentLoaded", async () => {
    await loadFilters();
});

// ------------------- LOGIN -------------------
loginButton.addEventListener("click", async () => {
    const username = usernameInput.value.trim();
    if (!username) return alert("Please enter a username");
    currentUser = username;

    // FIX: Removed userType logic. Backend handles new/existing user creation/lookup.

    try {
        // Update UI visibility
        loginSection.style.display = "none";
        recommendSection.style.display = "block";
        logoutButton.style.display = "block";

        // Load history/greeting. This handles both new/existing users.
        await loadHistory();
    } catch (err) {
        console.error("Login error:", err);
        alert("Failed to log in.");
    }
});

// ------------------- LOGOUT -------------------
logoutButton.addEventListener("click", () => {
    currentUser = "";
    loginSection.style.display = "block";
    recommendSection.style.display = "none";
    logoutButton.style.display = "none";
    historyDiv.innerHTML = "";
    recommendationsDiv.innerHTML = "";
});

// ------------------- FILTER OPTIONS -------------------
async function loadFilters() {
    await loadOptions("genre-options", "genres", "alpha");
    await loadOptions("movie-options", "movies", "rating");
}

async function loadOptions(selectId, endpoint, sortType) {
    const select = document.getElementById(selectId);
    select.innerHTML = "";
    try {
        const res = await fetch(`${API_BASE_URL}/${endpoint}`);

        if (!res.ok) {
            const errorDetail = await res
                .json()
                .catch(() => ({ detail: "Unknown error" }));
            throw new Error(
                `Failed to fetch ${endpoint} (${res.status}): ${errorDetail.detail}`
            );
        }

        let data = await res.json();

        if (sortType === "alpha") data.sort((a, b) => a.localeCompare(b));
        else if (sortType === "rating") data = data.slice(0, 25);

        data.forEach((item) => {
            const opt = document.createElement("option");
            opt.value = item;
            opt.textContent = item;
            select.appendChild(opt);
        });
        console.log(`✅ Loaded ${selectId.split("-")[0]} options.`);
    } catch (err) {
        console.error(`❌ Failed to load ${endpoint}:`, err);
        select.innerHTML = `<option disabled>Error loading data</option>`;
    }
}

// ------------------- GET RECOMMENDATIONS -------------------
document.getElementById("getRecsBtn").addEventListener("click", async () => {
    if (!currentUser) return alert("Please log in first");

    const type = document.getElementById("type").value;
    const genres = getSelectedValues("genre-options");
    const movies = getSelectedValues("movie-options");

    const payload = {
        username: currentUser,
        liked_genres: genres,
        liked_movies: movies,
        top_n: 10,
    };

    spinner.classList.add("show");
    recommendationsDiv.innerHTML = "";

    try {
        const res = await fetch(`${API_BASE_URL}/recommend/${type}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error("Failed to fetch recommendations");
        const data = await res.json();

        displayRecommendations(data.recommendations || []);
    } catch (err) {
        console.error("Recommendation error:", err);
        alert("Failed to fetch recommendations.");
    } finally {
        spinner.classList.remove("show");
    }
});

// ------------------- DISPLAY RECOMMENDATIONS -------------------
function displayRecommendations(recs) {
    recommendationsDiv.innerHTML = "";
    if (!recs.length) {
        recommendationsDiv.innerHTML = "<p>No recommendations found.</p>";
        return;
    }

    recs.forEach((m) => {
        const div = document.createElement("div");
        div.className = "movie-card";
        div.setAttribute("data-movie-id", m.movieId);
        div.innerHTML = `
            <span><strong>${m.title}</strong></span>
            <em>Rating: ${m.avg_rating?.toFixed(2) ?? "N/A"}</em>
            <em>Genres: ${m.genres.join(", ")}</em>
            <em>Tags: ${m.top_tags.join(", ")}</em>
            <div class="feedback-buttons">
                <button class="feedback-btn" data-type="interested">Interested</button>
                <button class="feedback-btn" data-type="watched">Watched and Liked</button>
                <button class="feedback-btn not-interested-btn" data-type="not_interested">Not Interested</button>
            </div>
        `;
        recommendationsDiv.appendChild(div);

        const [interestedBtn, watchedBtn, notInterestedBtn] =
            div.querySelectorAll(".feedback-btn");

        const setupFeedback = (button, type) => {
            button.addEventListener("click", async () => {
                const isActive = button.classList.contains("active");

                if (type === "not_interested") {
                    // Not Interested: Send permanent flag and hide card
                    await sendFeedback(m.movieId, "not_interested");
                    div.style.display = "none"; // Hide card immediately
                    await loadHistory(); // Update history panel
                    return;
                }

                // Interested/Watched Logic (Toggle)
                const action = isActive ? "remove" : type;

                try {
                    await sendFeedback(m.movieId, action);

                    // Update buttons state
                    button.classList.toggle("active", !isActive);

                    // FIX: Only update history. DO NOT re-run get recommendations.
                    await loadHistory();

                } catch (err) {
                    console.error("Feedback error:", err);
                }
            });
        };

        setupFeedback(interestedBtn, "interested");
        setupFeedback(watchedBtn, "watched");
        setupFeedback(notInterestedBtn, "not_interested");
    });
}

// Helper function to centralize feedback API call
async function sendFeedback(movieId, interaction) {
    if (!currentUser) return;
    try {
        const res = await fetch(`${API_BASE_URL}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: currentUser,
                movie_id: movieId,
                interaction: interaction,
            }),
        });
        if (!res.ok) throw new Error("Failed to send feedback");
    } catch (err) {
        console.error("Feedback API error:", err);
        alert(`Failed to record interaction: ${interaction}.`);
    }
}

// ------------------- LOAD HISTORY -------------------
async function loadHistory() {
    historyDiv.innerHTML = "";
    if (!currentUser) return;

    try {
        const res = await fetch(`${API_BASE_URL}/users/${currentUser}/history`);
        if (!res.ok) throw new Error("Failed to load history");
        const data = await res.json();

        if (data.is_new_user) {
            historyDiv.innerHTML = "<p>Welcome, new user!</p>";
            return;
        }

        displayHistory(data.history);
    } catch (err) {
        console.error("Failed to load history:", err);
    }
}

// ------------------- DISPLAY HISTORY -------------------
function displayHistory(history) {
    const historyContainer = document.getElementById("history");
    historyContainer.innerHTML = ""; // Clear the main history container

    if (!history || history.length === 0) {
        historyContainer.innerHTML = "<p>No interaction history yet.</p>";
        return;
    }

    // --- Section 1: Interested (Pending) ---
    const interestedDiv = document.createElement("div");
    interestedDiv.className = "history-group";
    interestedDiv.innerHTML = "<h4>👀 Movies You're Interested In</h4>";
    
    const interestedList = history.filter(h => h.interaction === 'interested');
    if (interestedList.length === 0) {
        interestedDiv.innerHTML += "<p>None marked interested yet.</p>";
    } else {
        interestedList.forEach((h) => {
            const div = document.createElement("div");
            div.className = "movie-card interested-card";
            div.innerHTML = `
                <span><strong>${h.title}</strong></span>
                <em>Genres: ${h.genres.join(", ")}</em>
            `;
            interestedDiv.appendChild(div);
        });
    }
    historyContainer.appendChild(interestedDiv);

    // --- Section 2: Watched and Liked (Completed) ---
    const watchedDiv = document.createElement("div");
    watchedDiv.className = "history-group";
    watchedDiv.innerHTML = "<h4>✅ Movies Watched and Liked</h4>";

    const watchedList = history.filter(h => h.interaction === 'watched');
    if (watchedList.length === 0) {
        watchedDiv.innerHTML += "<p>None marked watched yet.</p>";
    } else {
        watchedList.forEach((h) => {
            const div = document.createElement("div");
            div.className = "movie-card watched-card";
            div.innerHTML = `
                <span><strong>${h.title}</strong></span>
                <em>Genres: ${h.genres.join(", ")}</em>
            `;
            watchedDiv.appendChild(div);
        });
    }
    historyContainer.appendChild(watchedDiv);
}

// ------------------- UTIL -------------------
function getSelectedValues(selectId) {
    return Array.from(document.getElementById(selectId).selectedOptions).map(
        (opt) => opt.value
    );
}