export interface GenericResponse {
    // TODO: Update the response content
    something: string
}

export async function request(data: string): Promise<GenericResponse> {
    // TODO: Update the remote URL
    const res = await fetch("http://localhost:5000/api/v1", {
        method: "POST",
        body: JSON.stringify(data)
    })

    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
    }

    return res.json()
}
