const API_BASE_URL = "https://movie-recommender-backend-a9wo.onrender.com";
let currentUser = "";

// ------------------- DOM ELEMENTS -------------------
const loginSection = document.getElementById("login-section");
const recommendSection = document.getElementById("recommend-section");
const welcomeSection = document.getElementById("welcome-section");
const welcomeDiv = document.getElementById("welcome");
const informationSection = document.getElementById("information-section");
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

	try {
		// Update UI visibility
		welcomeDiv.textContent = `Welcome, ${currentUser}!`;
		loginSection.style.display = "none";
		welcomeSection.style.display = "block";
		informationSection.style.display = "block";
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
	welcomeSection.style.display = "none";
	recommendSection.style.display = "none";
	informationSection.style.display = "none";
	logoutButton.style.display = "none";
	historyDiv.innerHTML = "";
	recommendationsDiv.innerHTML = "";
});

// ------------------- FILTER OPTIONS -------------------
async function loadFilters() {
	await loadGenresCheckboxes();
	await loadMovieCheckboxes();
}

// --- Loads genres and creates checkboxes ---
async function loadGenresCheckboxes() {
	const container = document.getElementById("genre-checkboxes");
	if (!container) return;

	container.innerHTML = "";

	try {
		const res = await fetch(`${API_BASE_URL}/genres`);
		if (!res.ok) throw new Error("Failed to fetch genres");

		let genres = await res.json();

		genres.forEach((genre) => {
			const label = document.createElement("label");
			const checkbox = document.createElement("input");

			checkbox.type = "checkbox";
			checkbox.value = genre;
			checkbox.name = "selected-genre";

			label.appendChild(checkbox);
			label.appendChild(document.createTextNode(genre));
			container.appendChild(label);
		});

		console.log(`✅ Loaded genre checkboxes.`);
	} catch (err) {
		console.error(`❌ Failed to load genres:`, err);
		container.innerHTML = `<p style="color:red;">Error loading genres.</p>`;
	}
}

// --- Loads movies and creates checkboxes ---
async function loadMovieCheckboxes() {
	const container = document.getElementById("movie-checkboxes");
	if (!container) return;

	container.innerHTML = "";

	try {
		const res = await fetch(`${API_BASE_URL}/movies`);
		if (!res.ok) throw new Error("Failed to fetch top movies");

		let movies = await res.json();

		movies.forEach((movieTitle) => {
			const label = document.createElement("label");
			const checkbox = document.createElement("input");

			checkbox.type = "checkbox";
			checkbox.value = movieTitle;
			checkbox.name = "selected-movie";

			label.appendChild(checkbox);
			label.appendChild(document.createTextNode(movieTitle));
			container.appendChild(label);
		});

		console.log(`✅ Loaded movie checkboxes.`);
	} catch (err) {
		console.error(`❌ Failed to load top movies:`, err);
		container.innerHTML = `<p style="color:red;">Error loading top movies.</p>`;
	}
}

// ------------------- GET RECOMMENDATIONS -------------------
document.getElementById("getRecsBtn").addEventListener("click", async () => {
	if (!currentUser) return alert("Please log in first");

	const type = document.getElementById("type").value;

	// Get selected genres from the checkboxes
	const genreCheckboxes = document.querySelectorAll(
		'#genre-checkboxes input[name="selected-genre"]:checked'
	);
	const genres = Array.from(genreCheckboxes).map((cb) => cb.value);

	// Get selected movies from the checkboxes
	const movieCheckboxes = document.querySelectorAll(
		'#movie-checkboxes input[name="selected-movie"]:checked'
	);
	const movies = Array.from(movieCheckboxes).map((cb) => cb.value);

	const payload = {
		username: currentUser,
		liked_genres: genres,
		liked_movies: movies,
		top_n: 12,
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
	recommendationsDiv.classList.add("recommendation-grid");

	if (!recs.length) {
		recommendationsDiv.innerHTML = "<p>No recommendations found.</p>";
		recommendationsDiv.classList.remove("recommendation-grid");
		return;
	}

	recs.forEach((m) => {
		const div = document.createElement("div");
		div.className = "rec-card-poster";
		div.setAttribute("data-movie-id", m.movieId);

		// Use posterUrl if available, else use placeholder
		const placeholderUrl = `https://picsum.photos/200/300?random=${m.movieId}`;
		const posterUrl = m.posterUrl || placeholderUrl;

		div.innerHTML = `
            <div class="poster-container">
                <img src="${posterUrl}" alt="${m.title} Poster">
            </div>
            <div class="info-content">
                <span><strong>${m.title}</strong></span>
                <em>Rating: ${m.avg_rating?.toFixed(2) ?? "N/A"}</em>
                <em>Genres: ${m.genres.slice(0, 2).join(", ")}</em>
                <em>Tags: ${m.top_tags.slice(0, 1).join(", ")}</em>
            </div>
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
					// 1. Send permanent flag (for future exclusion)
					await sendFeedback(m.movieId, "not_interested");

					// 2. TOGGLE button appearance (light up red)
					button.classList.toggle("active", !isActive);

					// 3. Update history panel
					await loadHistory();

					return;
				}

				// Interested/Watched Logic (Toggle)
				const action = isActive ? "remove" : type;

				try {
					await sendFeedback(m.movieId, action);

					// Update buttons state
					button.classList.toggle("active", !isActive);

					// Only update history. DO NOT re-run get recommendations.
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
			historyDiv.innerHTML = "";
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

	const interestedList = history.filter((h) => h.interaction === "interested");
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

	const watchedList = history.filter((h) => h.interaction === "watched");
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
