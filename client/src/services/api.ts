export async function sendRaw(data: string): Promise<void> {
    // TODO: Update the remote URL
    const res = await fetch("http://localhost:5000/api/v1", {
        method: "POST",
        body: JSON.stringify(data)
    })

    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
    }
}

export interface ProcessedResponse {
    groundTruth: string,
    raw: string,
    preprocessed: string,
    processed: string,
    prompt: string
}

export async function getProcessed(): Promise<ProcessedResponse> {
    // TODO: Update the remote URL
    const res = await fetch("http://localhost:5000/api/v1", {
        method: "GET"
    })

    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
    }

    return res.json()
}
