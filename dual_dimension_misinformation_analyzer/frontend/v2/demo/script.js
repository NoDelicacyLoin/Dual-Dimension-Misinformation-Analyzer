////////////// Mock text section

const shortClaimText = "The Great Wall of China is visible from the Moon with the naked eye.";

const longClaimText = "I saw this claim in a short online post. The Great Wall of China is visible from the Moon with the naked eye and the Earth orbits the Sun.";

const greatWallSources = [
    {
        url: "https://www.britannica.com/question/Can-you-see-the-Great-Wall-of-China-from-space",
        content: "A popular myth, the claim was disproved when astronauts stated that the Great Wall of China was not visible with the naked eye from the Moon.",
        ai_analysis: "This evidence directly refutes the claim, stating it is a myth disproved by astronauts.",
        evidence_quality: "Strong",
        source_role: "Refutes Claim"
    },
    {
        url: "https://www.sciencefocus.com/planet-earth/is-the-great-wall-of-china-really-visible-from-space",
        content: "Contrary to popular belief, the Great Wall of China is a thin structure that blends in well with the surrounding landscape, making it very hard to spot.",
        ai_analysis: "This source explicitly states that the Great Wall is not visible from the Moon with the naked eye.",
        evidence_quality: "Strong",
        source_role: "Refutes Claim"
    },
    {
        url: "https://www.journalofoptometry.org/en-is-it-really-possible-see-articulo-S1888429608700542",
        content: "It would be totally impossible to see this cable, or for similar reasons the Great Wall, at a simple glance.",
        ai_analysis: "This evidence uses visual acuity calculations to explain why seeing the Great Wall from space would be impossible.",
        evidence_quality: "Usable",
        source_role: "Refutes Claim"
    }
];

const earthSources = [
    {
        url: "https://solarsystem.nasa.gov/planets/earth/overview/",
        content: "Earth makes a complete orbit around the Sun in about 365 days, defining the length of a year.",
        ai_analysis: "This source describes Earth's orbit around the Sun, so it supports the atomized claim.",
        evidence_quality: "Strong",
        source_role: "Supports Claim"
    },
    {
        url: "https://science.nasa.gov/earth/facts/",
        content: "Earth is the third planet from the Sun and travels around it as part of the solar system.",
        ai_analysis: "This source places Earth in orbit around the Sun and supports the claim.",
        evidence_quality: "Strong",
        source_role: "Supports Claim"
    }
];



////////////// Page element section

const textInput = document.querySelector("#text_input");
const rewriteCheckbox = document.querySelector("#rewrite_checkbox");
const shortTestButton = document.querySelector("#short_test_button");
const longTestButton = document.querySelector("#long_test_button");
const startAnalysisButton = document.querySelector("#start_button");
const returnHomeButton = document.querySelector("#return_home_button");

const inputPage = document.querySelector(".input_page");
const loadingPage = document.querySelector(".loading_page");
const resultPage = document.querySelector(".result_page");
const resultSummaryBlock = document.querySelector("#result_summary_block");
const resultClaimList = document.querySelector("#result_claim_list");

const moduleElements = {
    atomizer: {
        block: document.querySelector("#loading_module_atomizer"),
        circle: null
    },
    bert: {
        block: document.querySelector("#loading_module_bert"),
        circle: document.querySelector("#loading_circle_bert")
    },
    token: {
        block: document.querySelector("#loading_module_token_occlusion"),
        circle: document.querySelector("#loading_circle_token_occlusion")
    },
    tavily: {
        block: document.querySelector("#loading_module_tavily_nli"),
        circle: document.querySelector("#loading_circle_tavily_nli")
    },
    llm: {
        block: document.querySelector("#loading_module_llm"),
        circle: document.querySelector("#loading_circle_llm")
    },
    aggregation: {
        block: document.querySelector("#loading_module_aggregation"),
        circle: null
    }
};

let loadingTimer = null;



////////////// Input section

shortTestButton.addEventListener("click", function () {
    textInput.value = shortClaimText;
});

longTestButton.addEventListener("click", function () {
    textInput.value = longClaimText;
});

startAnalysisButton.addEventListener("click", function () {
    const inputText = textInput.value.trim();

    if (inputText.length === 0) {
        alert("Please enter a claim first.");
        return;
    }

    const loadingPlan = buildLoadingPlan(inputText, rewriteCheckbox.checked);

    showPage("loading");
    renderLoadingPlan(loadingPlan);
    startLoadingSimulation(loadingPlan);
});

returnHomeButton.addEventListener("click", function () {
    clearInterval(loadingTimer);
    showPage("input");
});



////////////// Page state section

function showPage(pageName) {
    inputPage.hidden = pageName !== "input";
    loadingPage.hidden = pageName !== "loading";
    resultPage.hidden = pageName !== "result";
}



////////////// Loading plan section

function buildLoadingPlan(inputText, rewriteEnabled) {
    const textUnits = splitTextIntoUnits(inputText);
    const factualClaims = [];
    let filteredUnits = 0;

    textUnits.forEach(function (textUnit) {
        if (isLikelyNonFactual(textUnit)) {
            filteredUnits = filteredUnits + 1;
            return;
        }

        const atomizedClaims = atomizeTextUnit(textUnit);

        atomizedClaims.forEach(function (claimText) {
            factualClaims.push(rewriteEnabled ? rewriteClaimText(claimText) : claimText.trim());
        });
    });

    if (factualClaims.length === 0) {
        factualClaims.push(rewriteEnabled ? rewriteClaimText(inputText) : inputText);
    }

    return {
        originalUnits: textUnits.length,
        filteredUnits: filteredUnits,
        totalClaims: factualClaims.length,
        claims: factualClaims,
        rewriteEnabled: rewriteEnabled
    };
}

function splitTextIntoUnits(inputText) {
    const cleanText = inputText.replace(/\n+/g, " ").trim();
    const matchedUnits = cleanText.match(/[^.!?]+[.!?]?/g);

    if (!matchedUnits) {
        return [];
    }

    return matchedUnits.map(function (unitText) {
        return unitText.trim();
    }).filter(function (unitText) {
        return unitText.length > 0;
    });
}

function isLikelyNonFactual(textUnit) {
    const lowerCaseUnit = textUnit.toLowerCase();
    const nonFactualMarkers = [
        "i think",
        "i feel",
        "i saw",
        "i heard",
        "awful",
        "disaster",
        "terrible",
        "amazing"
    ];

    const factualMarkers = [
        " is ",
        " are ",
        " was ",
        " were ",
        " causes ",
        " cause ",
        " alters ",
        " alter ",
        " orbits ",
        " exceeds ",
        " exceeded ",
        " visible ",
        " liquid "
    ];

    const hasNonFactualMarker = nonFactualMarkers.some(function (marker) {
        return lowerCaseUnit.includes(marker);
    });

    const hasFactualMarker = factualMarkers.some(function (marker) {
        return lowerCaseUnit.includes(marker);
    });

    return hasNonFactualMarker && !hasFactualMarker;
}

function atomizeTextUnit(textUnit) {
    const lowerCaseUnit = textUnit.toLowerCase();

    if (!lowerCaseUnit.includes(" and ")) {
        return [textUnit.trim()];
    }

    const splitParts = textUnit.split(/\band\b/i).map(function (partText) {
        return partText.trim().replace(/^[,.\s]+|[,.\s]+$/g, "");
    }).filter(function (partText) {
        return partText.length > 0;
    });

    if (splitParts.length < 2) {
        return [textUnit.trim()];
    }

    return splitParts;
}

function rewriteClaimText(claimText) {
    return claimText.replace(/\s+/g, " ").trim().replace(/\.$/, "");
}



////////////// Loading render section

function renderLoadingPlan(loadingPlan) {
    resetModuleProgress("atomizer");
    resetModuleProgress("bert");
    resetModuleProgress("token");
    resetModuleProgress("tavily");
    resetModuleProgress("llm");
    resetModuleProgress("aggregation");
}

function resetModuleProgress(moduleName) {
    const moduleElement = moduleElements[moduleName];

    if (moduleElement.circle) {
        moduleElement.circle.textContent = "0";
        moduleElement.circle.style.setProperty("--progress", "0deg");
    }

    moduleElement.block.classList.remove("module_active");
    moduleElement.block.classList.remove("module_done");
}

function updateModuleProgress(moduleName, completedClaims, totalClaims) {
    const moduleElement = moduleElements[moduleName];
    const progressDegree = Math.round((completedClaims / totalClaims) * 360);

    if (moduleElement.circle) {
        moduleElement.circle.textContent = completedClaims;
        moduleElement.circle.style.setProperty("--progress", progressDegree + "deg");
    }

    moduleElement.block.classList.remove("module_active");
    moduleElement.block.classList.remove("module_done");

    if (completedClaims > 0 && completedClaims < totalClaims) {
        moduleElement.block.classList.add("module_active");
    }

    if (completedClaims === totalClaims) {
        moduleElement.block.classList.add("module_done");
    }
}



////////////// Loading animation section

function startLoadingSimulation(loadingPlan) {
    const moduleOrder = ["bert", "tavily", "token", "llm", "aggregation"];
    const progressState = {
        atomizer: 0,
        bert: 0,
        token: 0,
        tavily: 0,
        llm: 0,
        aggregation: 0
    };

    let currentClaimIndex = 0;
    let currentModuleIndex = -1;

    clearInterval(loadingTimer);

    loadingTimer = setInterval(function () {
        if (progressState.atomizer < loadingPlan.totalClaims) {
            progressState.atomizer = progressState.atomizer + 1;
            updateModuleProgress("atomizer", progressState.atomizer, loadingPlan.totalClaims);
            return;
        }

        currentModuleIndex = currentModuleIndex + 1;

        if (currentModuleIndex >= moduleOrder.length) {
            currentModuleIndex = 0;
            currentClaimIndex = currentClaimIndex + 1;
        }

        if (currentClaimIndex >= loadingPlan.totalClaims) {
            clearInterval(loadingTimer);
            setTimeout(function () {
                showResultPage(loadingPlan);
            }, 500);
            return;
        }

        const currentModuleName = moduleOrder[currentModuleIndex];
        progressState[currentModuleName] = progressState[currentModuleName] + 1;
        updateModuleProgress(currentModuleName, progressState[currentModuleName], loadingPlan.totalClaims);
    }, 320);
}



////////////// Result section

function showResultPage(loadingPlan) {
    renderResultPage(loadingPlan);
    showPage("result");
}

function renderResultPage(loadingPlan) {
    const resultData = buildResultData(loadingPlan);

    resultSummaryBlock.innerHTML = makeOverallSummary(resultData, loadingPlan);
    resultClaimList.innerHTML = makeClaimBlocks(resultData);
}

function buildResultData(loadingPlan) {
    const inputText = textInput.value.trim();
    const factChecks = [];

    if (inputText.toLowerCase().includes("great wall")) {
        factChecks.push(makeGreatWallFactCheck(loadingPlan.rewriteEnabled));
    }

    if (inputText.toLowerCase().includes("earth orbits the sun")) {
        factChecks.push(makeEarthFactCheck(loadingPlan.rewriteEnabled));
    }

    return {
        original_claim: inputText,
        text_pattern: makeTextPatternResult(),
        fact_checks: factChecks,
        matrix_verdict: makeMatrixVerdict(factChecks)
    };
}

function makeGreatWallFactCheck(rewriteEnabled) {
    return {
        search_claim: "The Great Wall of China is visible from the Moon with the naked eye.",
        rewrite_is_enabled: rewriteEnabled,
        truth_score: 0.15,
        verdict: "False",
        decision_confidence: "high",
        evidence_sufficiency: "sufficient",
        sources: greatWallSources
    };
}

function makeEarthFactCheck(rewriteEnabled) {
    return {
        search_claim: "The Earth orbits the Sun.",
        rewrite_is_enabled: rewriteEnabled,
        truth_score: 0.95,
        verdict: "True",
        decision_confidence: "high",
        evidence_sufficiency: "sufficient",
        sources: earthSources
    };
}

function makeTextPatternResult() {
    return {
        prediction: {
            risk_level: "low_risk",
            risk_score: 0.3906,
            confidence_level: "low",
            probabilities: {
                low_risk: 0.5221,
                medium_risk: 0.1745,
                high_risk: 0.3034
            }
        },
        influential_words: [
            { text: "visible", confidence_drop: 0.3412 },
            { text: "naked", confidence_drop: 0.2875 },
            { text: "moon", confidence_drop: 0.2519 },
            { text: "earth", confidence_drop: 0.1784 }
        ],
        technical_details: {
            normalized_entropy: 0.9093
        },
        limitation: "This model checks wording patterns, not factual truth."
    };
}

function makeMatrixVerdict(factChecks) {
    const hasTrueFact = factChecks.some(function (factCheck) {
        return factCheck.verdict === "True";
    });

    const hasFalseFact = factChecks.some(function (factCheck) {
        return factCheck.verdict === "False";
    });

    if (hasTrueFact && hasFalseFact) {
        return {
            label: "Mixed",
            explanation: "This original text contains more than one factual claim. One is supported by evidence, while another is refuted."
        };
    }

    if (hasFalseFact) {
        return {
            label: "Refuted",
            explanation: "The text-pattern risk is low, but the evidence-based fact check refutes the factual claim."
        };
    }

    return {
        label: "Supported",
        explanation: "The evidence-based fact check supports the factual claim, and the text-pattern model does not show a strong risk signal."
    };
}

function makeOverallSummary(resultData, loadingPlan) {
    const evidenceScore = getEvidenceScore(resultData);
    const textRiskScore = getTextRiskScore(resultData.text_pattern);
    const textRiskLabel = getRiskLevelLabel(resultData.text_pattern.prediction.risk_level);
    const factVerdict = getFactVerdictLabel(resultData.fact_checks);
    const rewriteStatus = loadingPlan.rewriteEnabled ? "Enabled" : "Disabled";
    const originalClaimCount = loadingPlan.originalUnits;

    return (
        '<div class="overall_summary_main">' +
        '<p class="result_eyebrow">Overall Verdict</p>' +
        "<h2>" + escapeHtml(resultData.matrix_verdict.label) + "</h2>" +
        '<p class="summary_detail">' + originalClaimCount + " Claim(s) &middot; " + loadingPlan.filteredUnits + " Non-Factual Claim(s) Filtered</p>" +
        '<p class="summary_detail">' + loadingPlan.totalClaims + " Facts Being Checked &middot; Query Rewrite: " + rewriteStatus + "</p>" +
        "</div>" +
        '<div class="overall_summary_metrics">' +
        makeScoreRing("Overall Text-Pattern Risk", textRiskScore, textRiskLabel) +
        makeScoreRing("Overall Fact-Checking Support", evidenceScore, factVerdict) +
        "</div>"
    );
}

function makeScoreRing(title, score, label) {
    const displayScore = Math.floor(score);

    return (
        '<article class="score_card">' +
        '<div class="score_ring" style="--score: ' + score + '">' +
        '<div class="score_ring_inner">' +
        "<strong>" + displayScore + "%</strong>" +
        "<span>" + escapeHtml(label) + "</span>" +
        "</div>" +
        "</div>" +
        '<div class="score_card_text">' +
        "<h3>" + escapeHtml(title) + "</h3>" +
        "</div>" +
        "</article>"
    );
}

function makeClaimBlocks(resultData) {
    return (
        '<article class="claim_block">' +
        '<header class="claim_header">' +
        '<p class="result_eyebrow">Claim 1</p>' +
        "<h2>" + escapeHtml(resultData.original_claim) + "</h2>" +
        '<div class="claim_summary">' +
        "<span>Overall Verdict: " + escapeHtml(resultData.matrix_verdict.label) + "</span>" +
        "<span>Text-Pattern Risk: " + getTextRiskScore(resultData.text_pattern) + "%</span>" +
        "<span>Fact-Checking Support: " + getEvidenceScore(resultData) + "%</span>" +
        "</div>" +
        "</header>" +
        '<div class="claim_body">' +
        makeTextRiskPanel(resultData.text_pattern) +
        makeFactCheckingBlock(resultData) +
        "</div>" +
        '<footer class="matrix_note"><strong>Matrix Explanation:</strong> ' + escapeHtml(resultData.matrix_verdict.explanation) + "</footer>" +
        "</article>"
    );
}

function makeTextRiskPanel(textPattern) {
    return (
        '<section class="text_risk_panel">' +
        "<h3>Text-Pattern Risk</h3>" +
        '<div class="text_section">' +
        "<p><strong>Risk Score:</strong> " + formatPercent(textPattern.prediction.risk_score) + "</p>" +
        "<p><strong>Confidence Level:</strong> " + escapeHtml(getConfidenceDisplayLabel(textPattern.prediction.confidence_level)) + "</p>" +
        "<p><strong>Risk Level:</strong> " + escapeHtml(getRiskLevelLabel(textPattern.prediction.risk_level)) + "</p>" +
        "</div>" +
        '<div class="text_section">' +
        "<h4>Risk Distribution</h4>" +
        makeProbabilityRows(textPattern.prediction.probabilities) +
        "</div>" +
        '<div class="text_section">' +
        "<h4>Influential Words</h4>" +
        '<p class="small_note with_tooltip"><span>Showing the top 4 words by confidence drop.</span><span class="icon_tooltip" title="Confidence drop shows how much the model\'s confidence changed after masking one word.">?</span></p>' +
        makeTokenRows(textPattern.influential_words) +
        "</div>" +
        '<p class="small_note limitation_note">' + escapeHtml(textPattern.limitation) + "</p>" +
        "</section>"
    );
}

function makeFactCheckingBlock(resultData) {
    const factCheckingTitle = resultData.fact_checks.length > 1
        ? "Fact-Checking: Claim atomized into " + resultData.fact_checks.length + " factual claims"
        : "Fact-Checking: No Atomization Applied";

    return (
        '<section class="fact_checking_block">' +
        "<h3>" + escapeHtml(factCheckingTitle) + "</h3>" +
        resultData.fact_checks.map(function (factCheck, index) {
            return makeFactCheckingClaimBlock(factCheck, index);
        }).join("") +
        "</section>"
    );
}

function makeFactCheckingClaimBlock(factCheck, index) {
    const queryLabel = factCheck.rewrite_is_enabled ? "Rewrite Claim" : "Claim";
    const truthPercent = Math.round(factCheck.truth_score * 100);

    return (
        '<article class="fact_checking_claim_block">' +
        "<h4>Claim " + (index + 1) + ":</h4>" +
        '<section class="fact_checking_summary_each_claim">' +
        "<p><strong>" + queryLabel + ":</strong> " + escapeHtml(factCheck.search_claim) + "</p>" +
        '<div class="fact_checking_summary_tags">' +
        "<span>Verdict: " + escapeHtml(factCheck.verdict) + "</span>" +
        "<span>Truth Score: " + truthPercent + "%</span>" +
        "<span>Decision Confidence: " + escapeHtml(factCheck.decision_confidence) + "</span>" +
        "<span>Evidence Sufficiency: " + escapeHtml(factCheck.evidence_sufficiency) + "</span>" +
        "</div>" +
        "</section>" +
        makeEvidenceBlocks(factCheck.sources) +
        "</article>"
    );
}

function makeProbabilityRows(probabilities) {
    return Object.keys(probabilities).map(function (label) {
        const percent = Math.round(probabilities[label] * 100);

        return (
            '<div class="bar_row">' +
            "<span>" + escapeHtml(getRiskLevelLabel(label)) + "</span>" +
            '<span class="bar_track"><span class="bar_fill" style="width: ' + percent + '%"></span></span>' +
            "<span>" + percent + "%</span>" +
            "</div>"
        );
    }).join("");
}

function makeTokenRows(tokens) {
    return tokens.slice(0, 4).map(function (token) {
        return (
            '<div class="token_row">' +
            "<span>" + escapeHtml(token.text) + "</span>" +
            "<span>Confidence Drop: " + token.confidence_drop + "</span>" +
            "</div>"
        );
    }).join("");
}

function makeEvidenceBlocks(sources) {
    return sources.map(function (source) {
        const stanceClassName = getEvidenceStanceClass(source.source_role);

        return (
            '<article class="evidence_block">' +
            '<div class="evidence_tags">' +
            '<span class="evidence_tag evidence_stance ' + stanceClassName + '">Stance: ' + escapeHtml(source.source_role) + "</span>" +
            '<span class="evidence_tag evidence_quality">Evidence Quality: ' + escapeHtml(source.evidence_quality) + "</span>" +
            "</div>" +
            '<p><strong>Website:</strong> <a href="' + escapeHtml(source.url) + '" target="_blank">' + escapeHtml(source.url) + "</a></p>" +
            "<p><strong>Content:</strong> " + shortenText(source.content, 75) + "</p>" + "<br> "+
            '<p class="analysis_line"><strong>AI Analysis:</strong> ' + escapeHtml(source.ai_analysis) + "</p>" +
            "</article>"
        );
    }).join("");
}

function getEvidenceScore(resultData) {
    if (resultData.fact_checks.length === 0) {
        return 0;
    }

    let totalScore = 0;

    resultData.fact_checks.forEach(function (factCheck) {
        totalScore = totalScore + factCheck.truth_score;
    });

    return Math.round((totalScore / resultData.fact_checks.length) * 100);
}

function getTextRiskScore(textPattern) {
    return Math.round(textPattern.prediction.risk_score * 10000) / 100;
}

function getRiskLevelLabel(riskLevel) {
    if (riskLevel === "low_risk") {
        return "Low";
    }

    if (riskLevel === "medium_risk") {
        return "Medium";
    }

    if (riskLevel === "high_risk") {
        return "High";
    }

    return riskLevel;
}

function getConfidenceDisplayLabel(confidenceLevel) {
    if (confidenceLevel === "low") {
        return "Weak";
    }

    if (confidenceLevel === "medium") {
        return "Medium";
    }

    if (confidenceLevel === "high") {
        return "Strong";
    }

    return confidenceLevel;
}

function getFactVerdictLabel(factChecks) {
    if (factChecks.length === 0) {
        return "Neutral";
    }

    const hasTrueFact = factChecks.some(function (factCheck) {
        return factCheck.verdict === "True";
    });

    const hasFalseFact = factChecks.some(function (factCheck) {
        return factCheck.verdict === "False";
    });

    if (hasTrueFact && hasFalseFact) {
        return "Neutral";
    }

    if (hasFalseFact) {
        return "Refuted";
    }

    return "Supported";
}

function getEvidenceStanceClass(sourceRole) {
    const lowerCaseRole = sourceRole.toLowerCase();

    if (lowerCaseRole.includes("refute")) {
        return "stance_refutes";
    }

    if (lowerCaseRole.includes("support")) {
        return "stance_supports";
    }

    return "stance_neutral";
}

function formatPercent(numberValue) {
    return Math.round(numberValue * 10000) / 100 + "%";
}

function shortenText(text, maxLength) {
    if (text.length <= maxLength) {
        return escapeHtml(text);
    }

    return escapeHtml(text.slice(0, maxLength).trim()) + ".....";
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}
