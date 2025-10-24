// === Base URL (update this to match your Render backend) ===
const API_BASE_URL = window.location.hostname.includes("localhost")
	? "http://127.0.0.1:8000"
	: "https://movie-recommender-backend.onrender.com"; // ✅ adjust if your backend URL differs

let currentUsername = null;

// === Spinner control ===
function showSpinner(show) {
	const spinner = document.getElementById("spinner");
	if (show) {
		spinner.classList.add("show");
		spinner.style.display = "flex";
	} else {
		spinner.classList.remove("show");
		setTimeout(() => (spinner.style.display = "none"), 300);
	}
}

// === Populate dropdown options ===
async function populateOptions() {
	const GENRES = [
		"Action",
		"Adventure",
		"Animation",
		"Children",
		"Comedy",
		"Crime",
		"Documentary",
		"Drama",
		"Fantasy",
		"Film-Noir",
		"Horror",
		"Musical",
		"Mystery",
		"Romance",
		"Sci-Fi",
		"Thriller",
		"War",
		"Western",
	];
	const ACTORS = [
		"Nicolas Cage",
		"Tom Hanks",
		"Scarlett Johansson",
		"Leonardo DiCaprio",
		"Meryl Streep",
		"Brad Pitt",
		"Emma Watson",
		"Johnny Depp",
	];
	const MOVIES = [
		"Toy Story (1995)",
		"Jumanji (1995)",
		"Grumpier Old Men (1995)",
		"Waiting to Exhale (1995)",
		"Father of the Bride Part II (1995)",
		"Heat (1995)",
		"Sabrina (1995)",
		"GoldenEye (1995)",
		"Casino (1995)",
		"Sense and Sensibility (1995)",
		"Four Rooms (1995)",
		"Ace Ventura: When Nature Calls (1995)",
		"Money Train (1995)",
	];

	function fillSelect(id, list) {
		const select = document.getElementById(id);
		list.forEach((v) => {
			const opt = document.createElement("option");
			opt.value = v;
			opt.textContent = v;
			select.appendChild(opt);
		});
	}

	fillSelect("genre-options", GENRES);
	fillSelect("actor-options", ACTORS);
	fillSelect("movie-options", MOVIES);
}

// === Login ===
document.getElementById("loginBtn").addEventListener("click", async () => {
	const usernameInput = document.getElementById("username").value.trim();
	if (!usernameInput) return alert("Enter a username!");

	currentUsername = usernameInput.toLowerCase();
	localStorage.setItem("username", currentUsername);

	document.getElementById(
		"user-info"
	).textContent = `Welcome, ${currentUsername}!`;
	document.getElementById("recommend-section").style.display = "block";
	await loadHistory();
});

// === Get Recommendations ===
document.getElementById("getRecsBtn").addEventListener("click", async () => {
	if (!currentUsername) return alert("Please log in first!");

	const genres = Array.from(
		document.querySelectorAll("#genre-options option:checked")
	).map((o) => o.value);
	const actors = Array.from(
		document.querySelectorAll("#actor-options option:checked")
	).map((o) => o.value);
	const movies = Array.from(
		document.querySelectorAll("#movie-options option:checked")
	).map((o) => o.value);
	const type = document.getElementById("type").value;

	const container = document.getElementById("recommendations");
	container.innerHTML = "<em>Loading recommendations...</em>";
	showSpinner(true);

	try {
		const res = await fetch(`${API_BASE_URL}/recommend/${type}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				username: currentUsername,
				liked_genres: genres,
				liked_actors: actors,
				liked_movies: movies,
				top_n: 10,
			}),
		});

		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();

		container.innerHTML = "";
		data.recommendations.forEach((m) => {
			const div = document.createElement("div");
			div.className = "movie-card";
			div.innerHTML = `
				<span><strong>${m.title}</strong></span>
				<em>Rating: ${m.avg_rating?.toFixed(2) ?? "N/A"}</em>
				<em>Genres: ${m.genres.join(", ")}</em>
				<em>Tags: ${m.top_tags.join(", ")}</em>
				<div>
					<button class="feedback-btn" data-type="interested">Interested</button>
					<button class="feedback-btn" data-type="watched">Watched</button>
				</div>
			`;
			container.appendChild(div);

			const [interestedBtn, watchedBtn] = div.querySelectorAll(".feedback-btn");

			async function handleToggle(button, type) {
				const isActive = button.classList.contains("active");
				if (isActive) {
					button.classList.remove("active");
					button.textContent = type.charAt(0).toUpperCase() + type.slice(1);
					await sendFeedback(m.movieId, type, false);
				} else {
					button.classList.add("active");
					button.textContent =
						type === "interested" ? "Marked Interested" : "Marked Watched";
					await sendFeedback(m.movieId, type, true);
				}
				await loadHistory();
			}

			interestedBtn.addEventListener("click", () =>
				handleToggle(interestedBtn, "interested")
			);
			watchedBtn.addEventListener("click", () =>
				handleToggle(watchedBtn, "watched")
			);
		});
	} catch (e) {
		console.error("❌ Recommendation error:", e);
		container.innerHTML = "<em>Failed to load recommendations.</em>";
	} finally {
		showSpinner(false);
	}
});

// === Feedback API ===
async function sendFeedback(movieId, type, add = true) {
	if (!currentUsername) return;
	try {
		await fetch(`${API_BASE_URL}/feedback`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				username: currentUsername,
				movie_id: parseInt(movieId),
				interaction: add ? type : "remove",
			}),
		});
	} catch (e) {
		console.error("❌ Feedback error:", e);
	}
}

// === Load history ===
async function loadHistory() {
	if (!currentUsername) return;
	const div = document.getElementById("history");
	div.innerHTML = "<em>Loading...</em>";

	try {
		const res = await fetch(`${API_BASE_URL}/users/${currentUsername}/history`);
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();
		div.innerHTML = "";

		if (!data.history.length) {
			div.innerHTML = "<em>No history yet.</em>";
			return;
		}

		data.history.forEach((h) => {
			const item = document.createElement("div");
			item.className = "movie-card";
			item.innerHTML = `
				<span>${h.title}</span>
				<em>${h.interaction}</em>
				<em>Genres: ${h.genres.join(", ")}</em>
			`;
			div.appendChild(item);
		});
	} catch (e) {
		console.error("⚠️ Failed to load history:", e);
		div.innerHTML = "<em>No history yet.</em>";
	}
}

// === Logout ===
document.getElementById("logoutBtn").addEventListener("click", () => {
	localStorage.removeItem("username");
	currentUsername = null;
	document.getElementById("username").value = "";
	document.getElementById("recommend-section").style.display = "none";
	document.getElementById("user-info").textContent = "";
	document.getElementById("recommendations").innerHTML = "";
	document.getElementById("history").innerHTML = "";
});

// === On load ===
window.addEventListener("DOMContentLoaded", async () => {
	await populateOptions();

	const savedUser = localStorage.getItem("username");
	if (savedUser) {
		currentUsername = savedUser;
		document.getElementById("username").value = savedUser;
		document.getElementById(
			"user-info"
		).textContent = `Welcome back, ${savedUser}!`;
		document.getElementById("recommend-section").style.display = "block";
		await loadHistory();
	}
});
