import { StrictJSON } from "../lib/json"

const domain = "http://localhost:5000"

export async function handlePromiseState<T>(promise: Promise<T>, setData: (data: T) => void, setError: (error: string | null) => void): Promise<void> {
    try {
        const result = await promise
        setData(result)
    } catch (err: any) {
        setError(err.message)
    }
}

export async function sendRaw(data: string): Promise<void> {
    const parsed = StrictJSON.parse(data)
    const res = await fetch(`${domain}/data`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(parsed)
    })
    let body: any = null
    const contentType = res.headers.get("content-type")
    if (contentType && contentType.match(/text\/plain/)) {
        body = await res.text()
    }
    if (contentType && contentType.match(/application\/json/)) {
        body = await res.json()
    }
    if (body && !res.ok) {
        throw new Error(`${res.status} ${res.statusText} | ${body}`)
    }
    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
    }

    console.info(`${res.status} ${res.statusText} | ${body}`)
}

export async function getData(id: string): Promise<Record<string, any>> {
    const res = await fetch(`${domain}/data/${id}`, {
        method: "GET",
        headers: { "content-type": "application/json" },
    })
    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
    }

    return res.json()
}
