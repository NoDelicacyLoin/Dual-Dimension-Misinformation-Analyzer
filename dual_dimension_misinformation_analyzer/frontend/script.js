const shortClaimText = "The Great Wall of China is visible from the Moon with the naked eye.";
const longClaimText = "I saw this claim in a short online post. The Artemis program is led by NASA and aims to return humans to the Moon. The mission was delayed after technical problems. This pushed the planned launch into 2026.";

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
const originalInputPreviewLength = 170;

const displayVerdictLabels = {
    True: "Supported",
    "Mostly True": "Mostly Supported",
    False: "Refuted",
    "Mostly False": "Mostly Refuted"
};

const confidenceLevelLabels = {
    low: "Weak",
    medium: "Medium",
    high: "Strong"
};

const riskLevelLabels = {
    low_risk: "Low",
    medium_risk: "Medium",
    high_risk: "High"
};

const evidenceStanceLabels = {
    contradicts: "Refute",
    supports: "Support",
    background: "Background"
};

const evidenceStanceClasses = {
    contradicts: "stance_refutes",
    supports: "stance_supports",
    background: "stance_neutral"
};

const factualClaimStatusLabels = {
    no_evidence: "No Evidence Found",
    invalid_request: "Not Checkable",
    system_error: "System Error",
    degraded: "Degraded"
};

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

let activeRequestController = null;
const branchStepDelayMilliseconds = 420;
const resultPageDelayMilliseconds = 850;
let loadingProgressCounts = {
    textFeatureUnits: 0,
    factCheckClaims: 0
};

shortTestButton.addEventListener("click", function () {
    textInput.value = shortClaimText;
});

longTestButton.addEventListener("click", function () {
    textInput.value = longClaimText;
});

startAnalysisButton.addEventListener("click", function () {
    runAnalysis();
});

returnHomeButton.addEventListener("click", function () {
    cancelActiveRequest();
    showPage("input");
});

resultClaimList.addEventListener("click", function (event) {
    const toggleButton = event.target.closest("[data-original-text-toggle]");
    if (!toggleButton) {
        return;
    }

    const originalInputBlock = toggleButton.closest(".original_input_block");
    const textElement = originalInputBlock.querySelector("[data-original-text]");
    const fullText = originalInputBlock.dataset.fullText || "";
    const previewText = originalInputBlock.dataset.previewText || "";
    const isExpanded = originalInputBlock.dataset.expanded === "true";

    if (isExpanded) {
        originalInputBlock.dataset.expanded = "false";
        textElement.textContent = previewText;
        toggleButton.textContent = "Show Full Text";
    } else {
        originalInputBlock.dataset.expanded = "true";
        textElement.textContent = fullText;
        toggleButton.textContent = "Hide Full Text";
    }
});

async function runAnalysis() {
    const inputText = textInput.value.trim();

    if (!inputText) {
        alert("Please enter a claim first.");
        return;
    }

    const requestOptions = buildRequestOptions();
    const requestPayload = {
        claim: inputText,
        options: requestOptions
    };

    cancelActiveRequest();
    activeRequestController = new AbortController();

    startAnalysisButton.disabled = true;
    showPage("loading");
    resetLoadingModules();
    setActiveModules(["atomizer"]);

    try {
        const analysisResponse = await fetchStreamingAnalysis(requestPayload, activeRequestController.signal);
        await finishLoadingBeforeResult();
        renderResultPage(analysisResponse, requestOptions);
        showPage("result");
    } catch (error) {
        if (error.name === "AbortError") {
            return;
        }
        renderErrorPage(error, inputText);
        showPage("result");
    } finally {
        startAnalysisButton.disabled = false;
        activeRequestController = null;
    }
}

function buildRequestOptions() {
    return {
        use_query_rewrite: rewriteCheckbox.checked,
        relevance_threshold: 0.1,
        top_k: 3,
        use_all_eligible_evidence: false,
        retrieval_results: 10
    };
}

function cancelActiveRequest() {
    if (activeRequestController) {
        activeRequestController.abort();
        activeRequestController = null;
    }
}

function delay(milliseconds) {
    return new Promise(function (resolve) {
        window.setTimeout(resolve, milliseconds);
    });
}

function showPage(pageName) {
    inputPage.hidden = pageName !== "input";
    loadingPage.hidden = pageName !== "loading";
    resultPage.hidden = pageName !== "result";
}

async function fetchStreamingAnalysis(requestPayload, signal) {
    const response = await postAnalysisRequest("/analyze/stream", requestPayload, signal);

    if (!response.ok) {
        throw new Error(await getErrorMessageFromResponse(response));
    }

    if (!response.body) {
        return fetchAnalysisWithoutStreaming(requestPayload, signal);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let bufferedText = "";

    while (true) {
        const readResult = await reader.read();
        bufferedText += decoder.decode(readResult.value || new Uint8Array(), {
            stream: !readResult.done
        });

        const eventBlocks = bufferedText.split("\n\n");
        bufferedText = eventBlocks.pop() || "";

        for (const eventBlock of eventBlocks) {
            const streamEvent = parseStreamEvent(eventBlock);
            if (!streamEvent) {
                continue;
            }

            if (streamEvent.event === "progress") {
                updateLoadingFromProgress(streamEvent.data);
            } else if (streamEvent.event === "result") {
                return streamEvent.data;
            } else if (streamEvent.event === "error") {
                throw new Error(streamEvent.data.message || "The analysis failed.");
            }
        }

        if (readResult.done) {
            break;
        }
    }

    throw new Error("The analysis stream ended before returning a result.");
}

async function fetchAnalysisWithoutStreaming(requestPayload, signal) {
    const response = await postAnalysisRequest("/analyze", requestPayload, signal);

    if (!response.ok) {
        throw new Error(await getErrorMessageFromResponse(response));
    }

    return response.json();
}

function postAnalysisRequest(url, requestPayload, signal) {
    return fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(requestPayload),
        signal: signal
    });
}

async function getErrorMessageFromResponse(response) {
    try {
        const errorPayload = await response.json();
        if (errorPayload && errorPayload.detail) {
            return errorPayload.detail;
        }
    } catch (error) {
        // Fall through to the generic status message.
    }
    return "Analysis request failed with status " + response.status + ".";
}

function parseStreamEvent(eventBlock) {
    const dataLines = eventBlock
        .split("\n")
        .map(function (line) {
            return line.trim();
        })
        .filter(function (line) {
            return line.startsWith("data:");
        });

    if (dataLines.length === 0) {
        return null;
    }

    const jsonText = dataLines
        .map(function (line) {
            return line.slice(5).trim();
        })
        .join("\n");

    try {
        return JSON.parse(jsonText);
    } catch (error) {
        console.warn("Could not parse stream event:", jsonText);
        return null;
    }
}

function resetLoadingModules() {
    loadingProgressCounts = {
        textFeatureUnits: 0,
        factCheckClaims: 0
    };

    Object.keys(moduleElements).forEach(function (moduleName) {
        const moduleElement = moduleElements[moduleName];

        if (moduleElement.circle) {
            moduleElement.circle.textContent = "0";
            moduleElement.circle.style.setProperty("--progress", "0deg");
        }

        moduleElement.block.classList.remove("module_active");
        moduleElement.block.classList.remove("module_done");
    });
}

function setActiveModules(moduleNames) {
    moduleNames.forEach(function (moduleName) {
        if (!moduleElements[moduleName].block.classList.contains("module_done")) {
            moduleElements[moduleName].block.classList.add("module_active");
        }
    });
}

function setModuleProgress(moduleName, completedCount, totalCount) {
    const moduleElement = moduleElements[moduleName];
    if (!moduleElement.circle) {
        return;
    }

    const safeTotal = Number(totalCount) > 0 ? Number(totalCount) : 1;
    const safeCompleted = Math.max(0, Math.min(Number(completedCount) || 0, safeTotal));

    moduleElement.circle.textContent = safeCompleted + "/" + safeTotal;
    moduleElement.circle.style.setProperty("--progress", (safeCompleted / safeTotal * 360) + "deg");
}

function getModuleTotalCount(moduleName) {
    if (moduleName === "bert" || moduleName === "token") {
        return loadingProgressCounts.textFeatureUnits || 1;
    }

    if (moduleName === "tavily" || moduleName === "llm") {
        return loadingProgressCounts.factCheckClaims || 1;
    }

    return 1;
}

function markModuleDone(moduleName) {
    const moduleElement = moduleElements[moduleName];
    moduleElement.block.classList.remove("module_active");
    moduleElement.block.classList.add("module_done");

    if (moduleElement.circle) {
        const totalCount = getModuleTotalCount(moduleName);
        setModuleProgress(moduleName, totalCount, totalCount);
    }
}

function markAllModulesDone() {
    Object.keys(moduleElements).forEach(markModuleDone);
}

function markBranchModuleDone(moduleName) {
    markModuleDone(moduleName);
    showAggregationWhenBranchesAreDone();
}

function showAggregationWhenBranchesAreDone() {
    if (areBranchModulesDone()) {
        setActiveModules(["aggregation"]);
    }
}

function areBranchModulesDone() {
    return (
        moduleElements.bert.block.classList.contains("module_done") &&
        moduleElements.token.block.classList.contains("module_done") &&
        moduleElements.tavily.block.classList.contains("module_done") &&
        moduleElements.llm.block.classList.contains("module_done")
    );
}

function updateLoadingFromProgress(progressEvent) {
    if (!progressEvent || !progressEvent.stage) {
        return;
    }

    if (progressEvent.stage === "atomizer_finished") {
        loadingProgressCounts.textFeatureUnits = progressEvent.text_feature_unit_count || 0;
        loadingProgressCounts.factCheckClaims = progressEvent.fact_check_claim_count || 0;

        markModuleDone("atomizer");
        setModuleProgress("bert", 0, loadingProgressCounts.textFeatureUnits);
        setModuleProgress("token", 0, loadingProgressCounts.textFeatureUnits);
        setModuleProgress("tavily", 0, loadingProgressCounts.factCheckClaims);
        setModuleProgress("llm", 0, loadingProgressCounts.factCheckClaims);
        setActiveModules(["bert", "token", "tavily", "llm"]);
    } else if (progressEvent.stage === "bert_progress") {
        loadingProgressCounts.textFeatureUnits = progressEvent.text_feature_unit_count || loadingProgressCounts.textFeatureUnits;
        setModuleProgress("bert", progressEvent.completed_text_feature_unit_count || 0, loadingProgressCounts.textFeatureUnits);
    } else if (progressEvent.stage === "token_occlusion_progress") {
        loadingProgressCounts.textFeatureUnits = progressEvent.text_feature_unit_count || loadingProgressCounts.textFeatureUnits;
        setModuleProgress("token", progressEvent.completed_text_feature_unit_count || 0, loadingProgressCounts.textFeatureUnits);
    } else if (progressEvent.stage === "tavily_nli_progress") {
        loadingProgressCounts.factCheckClaims = progressEvent.fact_check_claim_count || loadingProgressCounts.factCheckClaims;
        setModuleProgress("tavily", progressEvent.completed_tavily_nli_count || 0, loadingProgressCounts.factCheckClaims);
    } else if (progressEvent.stage === "llm_evidence_progress") {
        loadingProgressCounts.factCheckClaims = progressEvent.fact_check_claim_count || loadingProgressCounts.factCheckClaims;
        setModuleProgress("llm", progressEvent.completed_llm_evidence_count || 0, loadingProgressCounts.factCheckClaims);
    } else if (progressEvent.stage === "text_pattern_finished") {
        loadingProgressCounts.textFeatureUnits = progressEvent.text_feature_unit_count || loadingProgressCounts.textFeatureUnits;
        markBranchModuleDone("bert");
        window.setTimeout(function () {
            markBranchModuleDone("token");
        }, branchStepDelayMilliseconds);
    } else if (progressEvent.stage === "fact_checking_finished") {
        loadingProgressCounts.factCheckClaims = progressEvent.fact_check_claim_count || loadingProgressCounts.factCheckClaims;
        markBranchModuleDone("tavily");
        window.setTimeout(function () {
            markBranchModuleDone("llm");
        }, branchStepDelayMilliseconds);
    } else if (progressEvent.stage === "analysis_finished") {
        markModuleDone("aggregation");
    }
}

async function finishLoadingBeforeResult() {
    if (!areBranchModulesDone()) {
        await delay(branchStepDelayMilliseconds + 80);
    }

    if (!areBranchModulesDone()) {
        markAllModulesDone();
    }

    markModuleDone("aggregation");
    await delay(resultPageDelayMilliseconds);
}

function renderResultPage(analysisResponse, requestOptions) {
    const claimGroups = buildClaimGroups(analysisResponse);

    resultSummaryBlock.innerHTML = makeOverallSummary(analysisResponse, requestOptions);
    resultClaimList.innerHTML = (
        makeOriginalInputBlock(analysisResponse.original_text) +
        makeClaimBlocks(analysisResponse, claimGroups)
    );
}

function renderErrorPage(error, originalText) {
    const safeMessage = escapeHtml(error.message || "The analysis failed.");
    const labels = getErrorPageLabels(error.message || "");
    resultSummaryBlock.innerHTML = (
        '<div class="overall_summary_main">' +
        '<p class="result_eyebrow">' + escapeHtml(labels.eyebrow) + "</p>" +
        "<h2>" + escapeHtml(labels.title) + "</h2>" +
        '<p class="summary_detail">' + safeMessage + "</p>" +
        "</div>" +
        '<div class="overall_summary_metrics">' +
        makeScoreRing("Overall Text-Pattern Risk", null, "N/A") +
        makeScoreRing("Overall Fact-Checking Support", null, "N/A") +
        "</div>"
    );
    resultClaimList.innerHTML = makeOriginalInputBlock(originalText);
}

function getErrorPageLabels(message) {
    const cleanMessage = String(message || "").toLowerCase();

    if (cleanMessage.includes("no checkable factual claim")) {
        return {
            eyebrow: "No checkable claim",
            title: "Unable to Analyze This Text"
        };
    }

    if (cleanMessage.includes("atomizer") || cleanMessage.includes("gemini")) {
        return {
            eyebrow: "Analysis system error",
            title: "Unable to Run Analysis"
        };
    }

    return {
        eyebrow: "Analysis error",
        title: "Unable to Analyze"
    };
}

function makeOverallSummary(analysisResponse, requestOptions) {
    const textRiskScore = getOverallTextRiskScore(analysisResponse.text_pattern_results || []);
    const factChecking = analysisResponse.fact_checking || {};
    const factTruthScore = toPercentScore(factChecking.truth_score);
    const factVerdictLabel = getDisplayVerdict(factChecking.verdict || getStatusLabel(factChecking.status || ""));
    const rewriteStatus = requestOptions.use_query_rewrite ? "On" : "Off";
    const detectedFactCount = analysisResponse.candidate_fact_claim_count || analysisResponse.fact_check_claim_count || 0;
    const checkedFactCount = analysisResponse.selected_fact_claim_count || analysisResponse.fact_check_claim_count || 0;

    return (
        '<div class="overall_summary_main">' +
        '<p class="result_eyebrow">Overall Verdict</p>' +
        "<h2>" + escapeHtml(factVerdictLabel || getStatusLabel(analysisResponse.status)) + "</h2>" +
        '<p class="summary_detail">' +
        "Detected: " + escapeHtml(detectedFactCount) + " factual &middot; " +
        "Filtered: " + escapeHtml(analysisResponse.ignored_sentence_count || 0) + " non-factual" +
        "</p>" +
        '<p class="summary_detail">' +
        "Checked: " + escapeHtml(checkedFactCount) + " factual &middot; " +
        "Rewrite: " + escapeHtml(rewriteStatus) +
        "</p>" +
        "</div>" +
        '<div class="overall_summary_metrics">' +
        makeScoreRing("Overall Text-Pattern Risk", textRiskScore, getRiskLevelLabelFromScore(textRiskScore)) +
        makeScoreRing("Overall Fact-Checking Support", factTruthScore, factVerdictLabel || "N/A") +
        "</div>"
    );
}

function makeOriginalInputBlock(originalText) {
    const cleanOriginalText = String(originalText || "").trim();
    if (!cleanOriginalText) {
        return "";
    }

    const previewText = shortenOriginalInput(cleanOriginalText);
    const isLongText = previewText !== cleanOriginalText;

    return (
        '<section class="original_input_block" data-expanded="false" data-full-text="' + escapeHtml(cleanOriginalText) + '" data-preview-text="' + escapeHtml(previewText) + '">' +
        '<div class="original_input_header">' +
        '<p class="result_eyebrow">User Input Text</p>' +
        (isLongText ? '<button type="button" data-original-text-toggle>Show Full Text</button>' : '') +
        '</div>' +
        '<p data-original-text>' + escapeHtml(previewText) + '</p>' +
        '</section>'
    );
}

function shortenOriginalInput(text) {
    if (text.length <= originalInputPreviewLength) {
        return text;
    }

    return text.slice(0, originalInputPreviewLength).trim() + "...";
}

function makeClaimBlocks(analysisResponse, claimGroups) {
    if (claimGroups.length === 0) {
        return (
            '<article class="claim_block">' +
            '<header class="claim_header">' +
            '<p class="result_eyebrow">No factual claim</p>' +
            "<h2>" + escapeHtml(analysisResponse.original_text || "No input text.") + "</h2>" +
            "</header>" +
            '<footer class="matrix_note">' + escapeHtml(analysisResponse.message || "No factual claim was available for analysis.") + "</footer>" +
            "</article>"
        );
    }

    return claimGroups.map(function (claimGroup, index) {
        return makeClaimBlock(claimGroup, index);
    }).join("");
}

function buildClaimGroups(analysisResponse) {
    const textPatternResults = analysisResponse.text_pattern_results || [];
    const factualClaims = (analysisResponse.fact_checking && analysisResponse.fact_checking.factual_claims) || [];
    const claimGroupMap = {};

    textPatternResults.forEach(function (textPatternResult) {
        const groupKey = String(textPatternResult.claim_group_id || "0");
        if (!claimGroupMap[groupKey]) {
            claimGroupMap[groupKey] = {
                claimGroupId: textPatternResult.claim_group_id || 0,
                originalSentence: textPatternResult.original_sentence || textPatternResult.text_feature_text || "",
                textPattern: textPatternResult,
                factualClaims: []
            };
        } else {
            claimGroupMap[groupKey].textPattern = textPatternResult;
            claimGroupMap[groupKey].originalSentence = claimGroupMap[groupKey].originalSentence || textPatternResult.original_sentence || "";
        }
    });

    factualClaims.forEach(function (factualClaim) {
        const groupKey = String(factualClaim.claim_group_id || "0");
        if (!claimGroupMap[groupKey]) {
            claimGroupMap[groupKey] = {
                claimGroupId: factualClaim.claim_group_id || 0,
                originalSentence: factualClaim.original_sentence || factualClaim.text_feature_text || factualClaim.claim || "",
                textPattern: null,
                factualClaims: []
            };
        }

        claimGroupMap[groupKey].factualClaims.push(factualClaim);
    });

    return Object.values(claimGroupMap).sort(function (left, right) {
        return left.claimGroupId - right.claimGroupId;
    });
}

function makeClaimBlock(claimGroup, index) {
    const textRiskScore = claimGroup.textPattern ? getTextRiskScore(claimGroup.textPattern) : null;
    const factTruthScore = getClaimGroupTruthScore(claimGroup.factualClaims);
    const factVerdict = getClaimGroupFactVerdict(claimGroup.factualClaims);

    return (
        '<article class="claim_block">' +
        '<header class="claim_header">' +
        '<p class="result_eyebrow">Claim ' + (index + 1) + "</p>" +
        "<h2>" + escapeHtml(claimGroup.originalSentence || "Untitled claim") + "</h2>" +
        '<div class="claim_summary">' +
        makeLabeledTag("Overall Verdict", factVerdict || "N/A") +
        makeLabeledTag("Text-Pattern Risk", formatScoreLabel(textRiskScore)) +
        makeLabeledTag("Fact-Checking Support", formatScoreLabel(factTruthScore)) +
        "</div>" +
        "</header>" +
        '<div class="claim_body">' +
        makeTextRiskPanel(claimGroup.textPattern) +
        makeFactCheckingBlock(claimGroup.factualClaims) +
        "</div>" +
        '<footer class="matrix_note"><strong>Matrix Explanation:</strong> ' + escapeHtml(makeMatrixExplanation(claimGroup)) + "</footer>" +
        "</article>"
    );
}

function makeMatrixExplanation(claimGroup) {
    if (claimGroup.factualClaims.length === 0) {
        return "The text-pattern branch returned a risk score, but no factual claim was available for evidence checking.";
    }

    const textRiskScore = getTextRiskScore(claimGroup.textPattern);
    const textRisk = typeof textRiskScore === "number"
        ? getRiskLevelLabelFromScore(textRiskScore)
        : "unavailable";
    const finalVerdicts = claimGroup.factualClaims
        .map(function (factualClaim) {
            return factualClaim.verdict;
        })
        .filter(Boolean);

    if (finalVerdicts.length === 0) {
        const statusLabel = getClaimGroupFactVerdict(claimGroup.factualClaims).toLowerCase();
        return "Evidence checking did not return a final verdict (" + statusLabel + "). Text-pattern risk is " + textRisk.toLowerCase() + ".";
    }

    const factVerdict = getClaimGroupFactVerdict(claimGroup.factualClaims);
    const lowerVerdict = factVerdict.toLowerCase();

    if (factVerdict === "Mixed") {
        return "Evidence is mixed. Treat this claim as uncertain; text-pattern risk is " + textRisk.toLowerCase() + ".";
    }

    if (lowerVerdict === "mostly supported") {
        if (textRisk === "High") {
            return "Evidence mostly supports the claim, but the wording pattern is high risk.";
        }
        return "Evidence mostly supports the claim; text-pattern risk is " + textRisk.toLowerCase() + ".";
    }

    if (lowerVerdict === "supported") {
        if (textRisk === "High") {
            return "Evidence supports the claim, but the wording pattern is high risk.";
        }
        if (textRisk === "Medium") {
            return "Evidence supports the claim, with some wording risk signals.";
        }
        return "Evidence supports the claim and the wording pattern is low risk.";
    }

    if (lowerVerdict === "mostly refuted") {
        if (textRisk === "High") {
            return "Evidence mostly refutes the claim, and the wording pattern is high risk.";
        }
        return "Evidence mostly refutes the claim; text-pattern risk is " + textRisk.toLowerCase() + ".";
    }

    if (lowerVerdict === "refuted") {
        if (textRisk === "High") {
            return "Both branches raise concern: evidence refutes the claim and the wording pattern is high risk.";
        }
        if (textRisk === "Medium") {
            return "Evidence refutes the claim, with medium wording risk signals.";
        }
        return "Evidence refutes the claim, even though the wording pattern looks low risk.";
    }

    if (lowerVerdict === "no evidence found") {
        return "Evidence checking found no selected evidence. Text-pattern risk is " + textRisk.toLowerCase() + ".";
    }

    if (lowerVerdict === "system error" || lowerVerdict === "degraded") {
        return "Evidence checking was unavailable. Text-pattern risk is " + textRisk.toLowerCase() + ".";
    }

    if (lowerVerdict === "not checkable") {
        return "This claim was not checkable by the evidence branch. Text-pattern risk is " + textRisk.toLowerCase() + ".";
    }

    return "Evidence verdict is " + lowerVerdict + "; text-pattern risk is " + textRisk.toLowerCase() + ".";
}

function makeTextRiskPanel(textPattern) {
    if (!textPattern) {
        return (
            '<section class="text_risk_panel">' +
            "<h3>Text-Pattern Risk</h3>" +
            '<p class="small_note">No text-pattern result was returned for this claim.</p>' +
            "</section>"
        );
    }

    if (textPattern.status !== "success") {
        return (
            '<section class="text_risk_panel">' +
            "<h3>Text-Pattern Risk</h3>" +
            '<p class="small_note">' + escapeHtml(textPattern.message || "Text-pattern analysis was unavailable for this claim.") + "</p>" +
            "</section>"
        );
    }

    const prediction = textPattern.prediction || {};

    return (
        '<section class="text_risk_panel">' +
        "<h3>Text-Pattern Risk</h3>" +
        '<div class="text_section">' +
        "<p><strong>Risk Score:</strong> " + formatScoreLabel(getTextRiskScore(textPattern)) + "</p>" +
        "<p><strong>Confidence Level:</strong> " + escapeHtml(getConfidenceDisplayLabel(prediction.confidence_level || "")) + "</p>" +
        "<p><strong>Risk Level:</strong> " + escapeHtml(getRiskLevelLabel(prediction.risk_level || "")) + "</p>" +
        "</div>" +
        '<div class="text_section">' +
        "<h4>Risk Distribution</h4>" +
        makeProbabilityRows(prediction) +
        "</div>" +
        '<div class="text_section">' +
        "<h4>Influential Words</h4>" +
        '<p class="small_note with_tooltip"><span>Showing the top words by confidence drop.</span><span class="icon_tooltip" title="Confidence drop shows how much the model confidence changed after masking one word.">?</span></p>' +
        makeTokenRows(textPattern.influential_words || []) +
        "</div>" +
        '<p class="small_note limitation_note">This model checks wording patterns, not factual truth.</p>' +
        "</section>"
    );
}

function makeFactCheckingBlock(factualClaims) {
    if (!factualClaims || factualClaims.length === 0) {
        return (
            '<section class="fact_checking_block">' +
            "<h3>Fact-Checking</h3>" +
            '<p class="small_note">No factual claim was returned for evidence checking.</p>' +
            "</section>"
        );
    }

    const title = factualClaims.length > 1
        ? "Fact-Checking: Multiple Checkable Claims"
        : "Fact-Checking: Single Checkable Claim";

    return (
        '<section class="fact_checking_block">' +
        "<h3>" + escapeHtml(title) + "</h3>" +
        factualClaims.map(function (factualClaim, index) {
            return makeFactualClaimBlock(factualClaim, index);
        }).join("") +
        "</section>"
    );
}

function makeFactualClaimBlock(factualClaim, index) {
    const truthScore = toPercentScore(factualClaim.truth_score);
    const evidenceItems = factualClaim.evidence || [];
    const verdictLabel = getFactualClaimVerdictLabel(factualClaim);
    const evidenceSufficiencyLabel = getEvidenceSufficiencyLabel(factualClaim);
    const decisionConfidenceLabel = getDecisionConfidenceLabel(factualClaim);
    const truthScoreLabel = typeof truthScore === "number" ? formatScoreLabel(truthScore) : "No score";

    return (
        '<article class="fact_checking_claim_block">' +
        "<h4>Claim " + (index + 1) + ":</h4>" +
        '<section class="fact_checking_summary_each_claim">' +
        "<p><strong>Checkable Claim:</strong> " + escapeHtml(factualClaim.claim || "") + "</p>" +
        '<div class="fact_checking_summary_tags">' +
        makeLabeledTag("Verdict", verdictLabel) +
        makeLabeledTag("Truth Score", truthScoreLabel) +
        makeLabeledTag("Confidence", decisionConfidenceLabel) +
        makeLabeledTag("Evidence Sufficiency", evidenceSufficiencyLabel) +
        "</div>" +
        "</section>" +
        makeEvidenceBlocks(evidenceItems) +
        "</article>"
    );
}

function makeEvidenceBlocks(evidenceItems) {
    if (!evidenceItems || evidenceItems.length === 0) {
        return '<p class="small_note">No selected evidence was returned for this factual claim.</p>';
    }

    return evidenceItems.map(function (evidenceItem) {
        const stance = evidenceItem.stance || "background";
        const stanceClassName = getEvidenceStanceClass(stance);

        return (
            '<article class="evidence_block">' +
            '<div class="evidence_tags">' +
            makeTag(getEvidenceStanceLabel(stance), "evidence_tag evidence_stance " + stanceClassName) +
            makeLabeledTag("Quality", getStatusLabel(evidenceItem.evidence_quality || "unknown"), "evidence_tag evidence_quality") +
            "</div>" +
            makeEvidenceUrlLine(evidenceItem.url) +
            "<p><strong>Content:</strong> " + shortenText(evidenceItem.content || "", 85) + "</p>" +
            '<p class="analysis_line"><strong>AI Analysis:</strong> ' + escapeHtml(evidenceItem.ai_analysis || "No source-level analysis was returned.") + "</p>" +
            "</article>"
        );
    }).join("");
}

function makeEvidenceUrlLine(url) {
    if (!url) {
        return "<p><strong>Website:</strong> N/A</p>";
    }

    return '<p><strong>Website:</strong> <a href="' + escapeHtml(url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(url) + "</a></p>";
}

function makeProbabilityRows(prediction) {
    const probabilities = {
        low_risk: prediction.low_risk_probability || 0,
        medium_risk: prediction.medium_risk_probability || 0,
        high_risk: prediction.high_risk_probability || 0
    };

    return Object.keys(probabilities).map(function (riskLevel) {
        const percent = Math.round(probabilities[riskLevel] * 100);

        return (
            '<div class="bar_row">' +
            "<span>" + escapeHtml(getRiskLevelLabel(riskLevel)) + "</span>" +
            '<span class="bar_track"><span class="bar_fill" style="width: ' + percent + '%"></span></span>' +
            "<span>" + percent + "%</span>" +
            "</div>"
        );
    }).join("");
}

function makeTokenRows(tokens) {
    if (!tokens || tokens.length === 0) {
        return '<p class="small_note">No influential words were returned.</p>';
    }

    return tokens.slice(0, 4).map(function (token) {
        return (
            '<div class="token_row">' +
            "<span>" + escapeHtml(token.text || "") + "</span>" +
            "<span>Confidence Drop: " + escapeHtml(Number(token.confidence_drop || 0).toFixed(4)) + "</span>" +
            "</div>"
        );
    }).join("");
}

function makeTag(text, className) {
    const classAttribute = className ? ' class="' + className + '"' : "";
    return "<span" + classAttribute + ">" + escapeHtml(text) + "</span>";
}

function makeLabeledTag(label, value, className) {
    return makeTag(label + ": " + value, className);
}

function makeScoreRing(title, score, label) {
    const safeScore = typeof score === "number" && Number.isFinite(score)
        ? Math.max(0, Math.min(100, score))
        : 0;
    const displayScore = typeof score === "number" && Number.isFinite(score)
        ? Math.round(safeScore) + "%"
        : "N/A";
    const labelClass = getScoreCardLabelClass(label);

    return (
        '<article class="score_card">' +
        '<div class="score_ring" style="--score: ' + safeScore + '">' +
        '<div class="score_ring_inner">' +
        "<strong>" + displayScore + "</strong>" +
        "</div>" +
        "</div>" +
        '<div class="score_card_text">' +
        "<h3>" + escapeHtml(title) + ":</h3>" +
        '<p class="score_card_label ' + labelClass + '">' + escapeHtml(label || "N/A") + "</p>" +
        "</div>" +
        "</article>"
    );
}

function getScoreCardLabelClass(label) {
    const cleanLabel = String(label || "").toLowerCase();

    if (cleanLabel === "low" || cleanLabel.includes("supported")) {
        return "score_label_good";
    }

    if (cleanLabel === "medium" || cleanLabel === "neutral" || cleanLabel === "mixed") {
        return "score_label_neutral";
    }

    if (cleanLabel === "high" || cleanLabel.includes("refuted")) {
        return "score_label_bad";
    }

    return "";
}

function getTextRiskScore(textPattern) {
    if (!textPattern || textPattern.status !== "success") {
        return null;
    }

    const prediction = textPattern.prediction || {};
    return toPercentScore(prediction.risk_score);
}

function getOverallTextRiskScore(textPatternResults) {
    const scores = textPatternResults
        .filter(function (textPatternResult) {
            return textPatternResult.status === "success";
        })
        .map(getTextRiskScore)
        .filter(function (score) {
            return typeof score === "number";
        });

    if (scores.length === 0) {
        return null;
    }

    return scores.reduce(function (total, score) {
        return total + score;
    }, 0) / scores.length;
}

function getClaimGroupTruthScore(factualClaims) {
    const scores = factualClaims
        .map(function (factualClaim) {
            return factualClaim.truth_score;
        })
        .filter(function (truthScore) {
            return typeof truthScore === "number";
        });

    if (scores.length === 0) {
        return null;
    }

    return toPercentScore(scores.reduce(function (total, score) {
        return total + score;
    }, 0) / scores.length);
}

function getClaimGroupFactVerdict(factualClaims) {
    if (!factualClaims || factualClaims.length === 0) {
        return "";
    }

    const verdicts = factualClaims
        .map(function (factualClaim) {
            return factualClaim.verdict;
        })
        .filter(Boolean);

    if (verdicts.length === 0) {
        return getFactualClaimVerdictLabel(factualClaims[0]);
    }

    const hasTrue = verdicts.some(function (verdict) {
        return verdict === "True" || verdict === "Mostly True";
    });
    const hasFalse = verdicts.some(function (verdict) {
        return verdict === "False" || verdict === "Mostly False";
    });

    if (hasTrue && hasFalse) {
        return "Mixed";
    }

    return getDisplayVerdict(verdicts[0]);
}

function getFactualClaimVerdictLabel(factualClaim) {
    if (factualClaim.verdict) {
        return getDisplayVerdict(factualClaim.verdict);
    }

    return factualClaimStatusLabels[factualClaim.status] || "No Final Verdict";
}

function getEvidenceSufficiencyLabel(factualClaim) {
    if (factualClaim.evidence_sufficiency) {
        return getStatusLabel(factualClaim.evidence_sufficiency);
    }

    if (factualClaim.status === "system_error" || factualClaim.status === "degraded") {
        return "Unavailable";
    }

    if (!factualClaim.evidence || factualClaim.evidence.length === 0) {
        return "No Selected Evidence";
    }

    return "Not Assessed";
}

function getDecisionConfidenceLabel(factualClaim) {
    if (factualClaim.status !== "success" && typeof factualClaim.truth_score !== "number") {
        if (factualClaim.status === "system_error" || factualClaim.status === "degraded") {
            return "Unavailable";
        }

        return "Not Assessed";
    }

    return getConfidenceDisplayLabel(factualClaim.decision_confidence || "") || "Not Assessed";
}

function toPercentScore(score) {
    if (typeof score !== "number" || !Number.isFinite(score)) {
        return null;
    }

    return Math.round(score * 10000) / 100;
}

function formatScoreLabel(score) {
    if (typeof score !== "number" || !Number.isFinite(score)) {
        return "N/A";
    }

    return Math.round(score * 100) / 100 + "%";
}

function getRiskLevelLabel(riskLevel) {
    return riskLevelLabels[riskLevel] || getStatusLabel(riskLevel);
}

function getRiskLevelLabelFromScore(score) {
    if (typeof score !== "number" || !Number.isFinite(score)) {
        return "N/A";
    }

    if (score >= 66) {
        return "High";
    }

    if (score >= 33) {
        return "Medium";
    }

    return "Low";
}

function getDisplayVerdict(verdict) {
    return displayVerdictLabels[verdict] || verdict || "";
}

function getConfidenceDisplayLabel(confidenceLevel) {
    return confidenceLevelLabels[confidenceLevel] || getStatusLabel(confidenceLevel);
}

function getEvidenceStanceClass(stance) {
    return evidenceStanceClasses[stance] || "stance_neutral";
}

function getEvidenceStanceLabel(stance) {
    return evidenceStanceLabels[stance] || getStatusLabel(stance);
}

function getStatusLabel(status) {
    if (!status) {
        return "";
    }

    return String(status)
        .split("_")
        .map(function (word) {
            return word.charAt(0).toUpperCase() + word.slice(1);
        })
        .join(" ");
}

function shortenText(text, maxLength) {
    const cleanText = String(text || "");

    if (cleanText.length <= maxLength) {
        return escapeHtml(cleanText);
    }

    return escapeHtml(cleanText.slice(0, maxLength).trim()) + "...";
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}
