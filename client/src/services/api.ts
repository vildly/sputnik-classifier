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
    const res = await fetch(`${domain}/data`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(data)
    })
    let body: string | null = null
    const contentType = res.headers.get("content-type")
    if (contentType && contentType.match(/text\/plain/)) {
        body = await res.text()
    }
    if (body && !res.ok) {
        throw new Error(`${res.status} ${res.statusText} | ${body}`)
    }
    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
    }

    console.info(`${res.status} ${res.statusText} | ${body}`)
}

// TODO: Add return interface here when I know the return type fully
export async function getData(id: string): Promise<any> {
    const res = await fetch(`${domain}/data/${id}`, {
        method: "GET",
        headers: { "content-type": "application/json" },
    })
    let body: string | null = null
    const contentType = res.headers.get("content-type")
    if (contentType && contentType.match(/text\/plain/)) {
        body = await res.text()
    }
    if (body && !res.ok) {
        throw new Error(`${res.status} ${res.statusText} | ${body}`)
    }
    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
    }

    return res.json()
}
