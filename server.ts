import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json({ limit: '35mb' }));
app.use(express.urlencoded({ extended: true, limit: '35mb' }));

// Initialize GoogleGenAI client with standard user agent header
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Fallback cascading models list as specified
const FALLBACK_MODELS = [
  'gemini-2.5-flash',
  'gemini-flash-latest',
  'gemini-3.1-flash-lite',
  'gemini-2.0-flash',
];

// Tool declaration: registerExpense
const registerExpenseTool = {
  functionDeclarations: [
    {
      name: 'registerExpense',
      description: 'Registra un gasto en la aplicación Mi Presupuesto Mensual a partir de una descripción en lenguaje natural o del análisis visual de una imagen de recibo/factura.',
      parameters: {
        type: Type.OBJECT,
        description: 'Datos estrictos requeridos para registrar el gasto.',
        properties: {
          amount: {
            type: Type.NUMBER,
            description: 'Monto del gasto en pesos colombianos (COP). Debe ser un número entero positivo, sin puntos ni signos de moneda. Por ejemplo, 45000 para $45.000 COP.',
          },
          date: {
            type: Type.STRING,
            description: 'Fecha del gasto en formato estricto YYYY-MM-DD. Si no se puede leer con exactitud o dice hoy, usar "2026-11-17".',
          },
          description: {
            type: Type.STRING,
            description: 'Descripción o concepto breve del gasto (ejemplo: "Almuerzo en Crepes & Waffles", "Mercado en D1", "Pasaje TransMilenio", "Recibo de energía Enel", "Compras Éxito").',
          },
          category: {
            type: Type.STRING,
            description: 'Categoría obligatoria del gasto. Debe ser exactamente una de estas 5: "Alimentación", "Transporte", "Servicios públicos", "Ropa y compras personales", "Entretenimiento y ocio".',
          },
        },
        required: ['amount', 'date', 'description', 'category'],
      },
    },
  ],
};

// API Endpoint for the AI Assistant (Supports text and image multimodal input)
app.post('/api/assistant/chat', async (req, res) => {
  try {
    const {
      message = '',
      image = null,
      history = [],
      currentBudget = 3500000,
      currentSpent = 0,
      currentAvailable = 3500000,
    } = req.body;

    const hasMessage = Boolean(message && typeof message === 'string' && message.trim().length > 0);
    const hasImage = Boolean(image && image.data);

    if (!hasMessage && !hasImage) {
      res.status(400).json({ error: 'Debes enviar un mensaje de texto o adjuntar una imagen.', code: 400 });
      return;
    }

    if (!process.env.GEMINI_API_KEY) {
      res.status(500).json({
        error: 'No se encontró la variable GEMINI_API_KEY en el servidor.',
        code: 500,
      });
      return;
    }

    // System instruction: Brief, friendly, expert in personal finance in Colombia with Multimodal Vision
    const systemInstruction = `Eres el asistente inteligente de la aplicación "Mi Presupuesto Mensual" para usuarios en Colombia.
Tu objetivo es ayudar al usuario a registrar gastos y responder consultas sobre su presupuesto, tanto a través de mensajes de texto como analizando imágenes de recibos, facturas, tirillas de caja, comprobantes de pago digitales (Nequi, Daviplata, etc.) o fotos de compras.

Información financiera actual del usuario en Noviembre de 2026:
- Fecha actual: 17 de Noviembre de 2026 (2026-11-17).
- Presupuesto mensual total: $${currentBudget.toLocaleString('es-CO')} COP
- Total gastado hasta hoy: $${currentSpent.toLocaleString('es-CO')} COP
- Saldo disponible restante: $${currentAvailable.toLocaleString('es-CO')} COP

Reglas y Capacidades:
1. Análisis de Imágenes (Visión Multimodal):
   - Cuando el usuario envíe una imagen de una factura, recibo, tirilla de compra o comprobante bancario:
     * Lee cuidadosamente el nombre del comercio o establecimiento (ej: Éxito, D1, Ara, Carulla, Crepes & Waffles, Farmatodo, Terpel, TransMilenio, etc.).
     * Extrae el valor total pagado en pesos colombianos (COP).
     * Identifica o asigna la fecha adecuada (si no está clara o es reciente, usa "2026-11-17").
     * Determina la categoría más apropiada (DEBE ser exactamente una de estas 5: "Alimentación", "Transporte", "Servicios públicos", "Ropa y compras personales", "Entretenimiento y ocio").
     * Llama a la función 'registerExpense' para que el gasto quede registrado automáticamente en el Dashboard.
     * En tu respuesta textual, felicita amablemente al usuario por controlar sus gastos, resume lo detectado en la imagen y menciona su nuevo saldo disponible.

2. Gastos por Mensaje de Texto:
   - Si el usuario dice que gastó o pagó dinero (ej: "Gasté 50.000 en comida hoy", "Pagué 14.000 de TransMilenio", "128.000 de la luz"), DEBES llamar a 'registerExpense' con los 4 parámetros requeridos: amount (número), date (YYYY-MM-DD), description (string) y category.

3. Respuestas Generales:
   - Sé breve, cálido, empático y experto en finanzas personales en Colombia.
   - Si el usuario solo saluda o pregunta por su saldo disponible o recomendaciones, responde cordialmente en 1 o 2 oraciones citando sus números reales sin llamar a la función.`;

    // Build contents structure
    const contents: Array<{ role: 'user' | 'model'; parts: Array<any> }> = [];

    // Append prior history if available (text only for historical turns to save bandwidth)
    if (Array.isArray(history)) {
      for (const h of history.slice(-6)) {
        if (h.role === 'user' || h.role === 'model') {
          contents.push({
            role: h.role,
            parts: [{ text: String(h.text || '') }],
          });
        }
      }
    }

    // Build current user message parts (multimodal: image + text)
    const userParts: Array<any> = [];

    if (hasImage) {
      let cleanBase64 = String(image.data);
      let mimeType = image.mimeType || 'image/jpeg';
      if (cleanBase64.includes(';base64,')) {
        const parts = cleanBase64.split(';base64,');
        mimeType = parts[0].replace(/^data:/, '') || mimeType;
        cleanBase64 = parts[1];
      }

      userParts.push({
        inlineData: {
          mimeType,
          data: cleanBase64,
        },
      });
    }

    const promptText = hasMessage
      ? message.trim()
      : 'Por favor analiza la imagen de este recibo, comprobante de pago o factura. Extrae el monto total, fecha, comercio y categoría, y llama a la función registerExpense para registrarlo en mi presupuesto.';

    userParts.push({ text: promptText });

    // Append current user message
    contents.push({
      role: 'user',
      parts: userParts,
    });

    let successResponse: any = null;
    let modelUsed = '';
    let lastError: any = null;
    let lastStatusCode = 500;

    // Fallback Cascade Loop
    for (const model of FALLBACK_MODELS) {
      try {
        console.log(`[Assistant] Trying model in cascade: ${model}`);
        const response = await ai.models.generateContent({
          model,
          contents,
          config: {
            systemInstruction,
            tools: [registerExpenseTool],
          },
        });

        successResponse = response;
        modelUsed = model;
        console.log(`[Assistant] Model ${model} succeeded.`);
        break;
      } catch (err: any) {
        lastError = err;
        const errMsg = err?.message || String(err);
        const status = err?.status || err?.statusCode || (errMsg.includes('429') ? 429 : errMsg.includes('503') ? 503 : errMsg.includes('404') ? 404 : 500);
        lastStatusCode = status;
        console.warn(`[Assistant] Model ${model} failed with code ${status}: ${errMsg}. Falling back to next model...`);
      }
    }

    // If all models in the cascade failed
    if (!successResponse) {
      const errMsg = lastError?.message || 'Error al comunicarse con la API de Gemini.';

      if (lastStatusCode === 503 || errMsg.includes('503') || errMsg.includes('unavailable') || errMsg.includes('high demand')) {
        res.status(503).json({
          error: 'El asistente está muy solicitado en este momento. Por favor, intenta de nuevo en un minuto.',
          code: 503,
          technical: errMsg,
        });
        return;
      }

      if (lastStatusCode === 429 || errMsg.includes('429') || errMsg.includes('RESOURCE_EXHAUSTED') || errMsg.includes('quota')) {
        res.status(429).json({
          error: 'He alcanzado mi límite de mensajes por ahora. Intenta de nuevo más tarde.',
          code: 429,
          technical: errMsg,
        });
        return;
      }

      res.status(lastStatusCode >= 400 && lastStatusCode < 600 ? lastStatusCode : 500).json({
        error: errMsg,
        code: lastStatusCode || 500,
        technical: errMsg,
      });
      return;
    }

    // Robust Data Extraction: Manually inspect response.candidates[0].content.parts
    const candidate = successResponse.candidates?.[0];
    const parts = candidate?.content?.parts || [];

    let extractedText = '';
    const functionCalls: Array<{ name: string; args: any }> = [];

    for (const part of parts) {
      if (typeof part.text === 'string' && part.text) {
        extractedText += part.text;
      }
      if (part.functionCall) {
        functionCalls.push({
          name: part.functionCall.name,
          args: part.functionCall.args,
        });
      }
    }

    // Fallbacks if direct parts didn't extract
    if (!extractedText && successResponse.text) {
      extractedText = successResponse.text;
    }
    if (functionCalls.length === 0 && Array.isArray(successResponse.functionCalls)) {
      for (const fc of successResponse.functionCalls) {
        functionCalls.push({
          name: fc.name,
          args: fc.args,
        });
      }
    }

    // Check if registerExpense function was called
    const registerExpenseCall = functionCalls.find((fc) => fc.name === 'registerExpense');

    res.json({
      text: extractedText.trim() || (registerExpenseCall ? 'He registrado el gasto solicitado en tu presupuesto.' : '¿En qué más te puedo ayudar con tu presupuesto?'),
      functionCall: registerExpenseCall || null,
      modelUsed,
    });
  } catch (globalErr: any) {
    console.error('[Assistant] Unhandled error in /api/assistant/chat:', globalErr);
    res.status(500).json({
      error: globalErr?.message || 'Error interno del servidor en el asistente.',
      code: 500,
      technical: String(globalErr),
    });
  }
});

// Setup Vite dev server or static files in production
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`Mi Presupuesto Mensual full-stack server running on http://0.0.0.0:${port}`);
  });
}

startServer();
